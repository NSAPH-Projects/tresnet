import numpy as np
import torch
import argparse
from collections import defaultdict, deque
import pandas as pd
import os
import utils.ratios as ratios
import yaml

from models.DensityNet import DensityNet
from dataset.dataset import get_iter, make_dataset, DATASETS, set_seed


def main(args: argparse.Namespace) -> None:
    # seed and use gpu when available
    set_seed(1234 + 131 * args.seed)  # like torch manual seed at all levels
    dev = torch.cuda if torch.cuda.is_available() else "cpu"

    # these are the shifting values use in the srf curve
    steps = 10
    delta_list = torch.linspace(0.5 / steps, 0.5, steps=steps)

    # make dataset from available options
    D = make_dataset(args.dataset, delta_list, n_train=args.n_train, n_test=args.n_test)
    train_matrix = D["train_matrix"]
    shift_type = D["shift_type"]
    n, input_dim = train_matrix.shape[0], train_matrix.shape[1] - 2

    # make neural network model
    density_estimator_config = [(input_dim, 50, 1), (50, 50, 1)]
    model = DensityNet(density_estimator_config, args.num_grid).to(dev)
    density_head = model.density_estimator
    model._initialize_weights()
    optim_params = [{"params": model.parameters(), "weight_decay": args.wd}]

    # make regularizing layers if required
    if args.combo_reg:
        args.var_reg = True
        args.ratio_reg = True

    if args.var_reg:
        var_reg = ratios.VarianceRegularizer(delta_list)
        optim_params.append({"params": var_reg.parameters()})

    if args.ratio_reg:
        ratio_reg = ratios.RatioRegularizer(delta_list)
        optim_params.append({"params": ratio_reg.parameters()})

    # make optimizer
    if args.opt == "adam":
        optimizer = torch.optim.Adam(optim_params, lr=0.001)
    elif args.opt == "sgd":
        optimizer = torch.optim.SGD(optim_params, lr=0.0001, nesterov=True)

    # training loop
    train_loader = get_iter(train_matrix, batch_size=500, shuffle=False)

    for epoch in range(args.n_epoch):
        # dict to store all the losses per batch
        losses = defaultdict(lambda: deque(maxlen=len(train_loader)))

        # iterate each batch
        for idx, item in enumerate(train_loader):
            total_loss = torch.tensor(0.0, device=dev)

            # move tensor to gpu if available
            t = item["treatment"].to(dev)
            x = item["covariates"].to(dev)
            y = item["outcome"].to(dev)

            # zero grad and evaluate model
            optimizer.zero_grad()
            model_output = model.forward(t, x)
            probs = model_output["prob_score"]
            z = model_output["z"]

            # 1. negative loglikelihood loss
            density_negll = -probs.log().mean()
            total_loss = total_loss + density_negll
            losses["density_negll"].append(density_negll.item())

            # 2. regularization losses
            if args.var_reg:
                var_reg_loss = var_reg.loss(t, density_head, z, shift_type)
                var_reg_loss = var_reg_loss + (1 / n) * var_reg.prior()
                total_loss = total_loss + var_reg_loss
                losses["var_reg_loss"].append(var_reg_loss.item())

            if args.ratio_reg:
                ratio_reg_loss = ratio_reg.loss(t, density_head, z, shift_type)
                ratio_reg_loss = ratio_reg_loss + (1 / n) * ratio_reg.prior()
                total_loss = total_loss + ratio_reg_loss
                losses["ratio_reg_loss"].append(ratio_reg_loss.item())

            losses["total_loss"].append(total_loss.item())

            total_loss.backward()
            optimizer.step()

        # evaluation
        if epoch == 0 or (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                if not args.silent:
                    print("== Epoch: ", epoch, " ==")
                    print("Metrics:")
                    for k, vec in losses.items():
                        print(f"  {k}: {np.mean(vec):.4f}")

                # iptw estimates
                df = pd.DataFrame({"delta": delta_list})
                for part in ("train", "test"):
                    M = D[part + "_matrix"].to(dev)
                    t, x, y = M[:, 0], M[:, 1:-1], M[:, -1]
                    z = model.forward(t, x)["z"]
                    srf = D["srf_" + part]
                    df[part + "_truth"] = srf

                    estims = {}
                    errors = {}
                    shift_type = D["shift_type"]

                    for d, truth in zip(delta_list, srf):
                        log_ratio = ratios.log_density_ratio_under_shift(
                            treatment=t,
                            delta=d,
                            density_estimator=density_head,
                            z=z,
                            shift_type=shift_type,
                        )
                        ratio = log_ratio.exp()
                        ipw_estim = (ratio / ratio.mean() * y).mean().item()
                        error = (ipw_estim - truth).abs().item()

                        # make d as string key just for easy printing of results
                        key = np.round(d.item(), 3)
                        estims[key] = ipw_estim
                        errors[key] = error

                    # add estimation error as columns of result dataframe
                    df[part + "_estim"] = list(estims.values())
                    df[part + "_error"] = list(errors.values())
                    curve_error = np.mean(df[part + "_error"] ** 2) ** 0.5

                    if not args.silent:
                        print(f"SRF {part}:")
                        print(f"  @0.1: {errors[0.1]:.4f}")
                        print(f"  @0.5: {errors[0.5]:.4f}")
                        print(f"  curve: {curve_error:.4f}")

                # experiment_dir]
                s1 = f"var_{int(args.var_reg)}"
                s2 = f"ratio_{int(args.ratio_reg)}"
                exp_dir = f"{'-'.join([s1, s2])}/{args.seed:02d}"

                # save estimated curve dataset
                results_path = f"results/{args.dataset}/{exp_dir}/curve.csv"
                os.makedirs(os.path.dirname(results_path), exist_ok=True)
                df.round(4).to_csv(results_path, index=False)

                # save experiment config
                config_path = f"results/{args.dataset}/{exp_dir}/config.yaml"
                with open(config_path, "w") as io:
                    yaml.safe_dump(vars(args), io)

                # save metrics
                metrics_path = f"results/{args.dataset}/{exp_dir}/metrics.yaml"
                metrics = {k: float(np.mean(v)) for k, v in losses.items()}
                metrics["last_saved_epoch"] = epoch
                metrics["@0.1"] = float(errors[0.1])
                metrics["@0.5"] = float(errors[0.5])
                metrics["curve"] = float(curve_error)
                metrics["seed"] = args.seed
                with open(metrics_path, "w") as io:
                    yaml.safe_dump(metrics, io)

            model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_grid", default=10, type=int)
    parser.add_argument("--dataset", default="ihdp-N", type=str, choices=DATASETS)
    parser.add_argument("--opt", default="adam", type=str, choices=("adam", "sgd"))
    parser.add_argument("--n_train", default=500, type=int)
    parser.add_argument("--n_test", default=200, type=int)
    parser.add_argument("--n_epoch", default=10000, type=int)
    parser.add_argument("--wd", default=1e-6, type=int)
    parser.add_argument("--silent", default=False, action="store_true")

    # regularizations available
    parser.add_argument("--var_reg", default=False, action="store_true")
    parser.add_argument("--ratio_reg", default=False, action="store_true")
    parser.add_argument("--combo_reg", default=False, action="store_true")

    args = parser.parse_args()
    main(args)
