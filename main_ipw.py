import numpy as np
import torch
import argparse
from collections import defaultdict, deque
import pandas as pd
import os
import utils.ratios as ratios
import yaml
import matplotlib.pyplot as plt

plt.ioff()

from models.DensityNet import DensityNet
from dataset.dataset import get_iter, make_dataset, DATASETS, set_seed


def main(args: argparse.Namespace) -> None:
    # seed and use gpu when available
    set_seed(1234 + 131 * args.seed)  # like torch manual seed at all levels
    dev = torch.cuda if torch.cuda.is_available() else "cpu"

    # these are the shifting values use in the srf curve
    delta_steps = 10
    delta_list = torch.linspace(0.5 / delta_steps, 0.5, steps=delta_steps)

    # make dataset from available options
    D = make_dataset(args.dataset, delta_list, n_train=args.n_train, n_test=args.n_test)
    train_matrix = D["train_matrix"]
    shift_type = D["shift_type"]
    n, input_dim = train_matrix.shape[0], train_matrix.shape[1] - 2

    # make neural network model
    density_estimator_config = [(input_dim, 50, 1), (50, 50, 1)]
    model = DensityNet(
        density_estimator_config, args.n_grid, dropout=args.dropout
    ).to(dev)
    density_head = model.density_estimator
    model._initialize_weights()
    optim_params = [{"params": model.parameters(), "weight_decay": args.wd}]

    # set regularizations if other flags require it
    if args.combo_reg:
        args.var_reg = True
        args.ratio_reg = True

    if not args.erm:
        args.var_reg = True

    # make regularizing layers if required
    if args.var_reg:
        var_reg = ratios.VarianceRegularizer(
            delta_list=delta_list, multiscale=args.reg_multiscale
        )
        optim_params.append({"params": var_reg.parameters()})

    if args.ratio_reg:
        ratio_reg = ratios.RatioRegularizer(
            delta_list=delta_list,
            multiscale=args.reg_multiscale,
            fit_scale=args.fit_ratio_scale,
        )
        optim_params.append({"params": ratio_reg.parameters()})

    if args.pos_reg:
        pos_reg = ratios.PosteriorRegularizer(
            delta_list=delta_list,
            multiscale=args.reg_multiscale,
        )
        optim_params.append({"params": pos_reg.parameters()})

    # make optimizer
    if args.opt == "adam":
        optimizer = torch.optim.Adam(optim_params, lr=0.001)
    elif args.opt == "sgd":
        optimizer = torch.optim.SGD(
            optim_params, lr=0.0001, momentum=0.9, nesterov=True
        )

    # training loop
    train_loader = get_iter(train_matrix, batch_size=args.batch_size, shuffle=False)

    for epoch in range(args.n_epochs):
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
            density_negll = - probs.log().mean()
            losses["density_negll"].append(density_negll.item())
            if args.erm:
                total_loss = total_loss + density_negll

            # 2. regularization losses
            if args.var_reg:
                var_reg_loss = var_reg(t, density_head, z, shift_type)
                var_reg_loss = var_reg_loss + (1 / n) * var_reg.prior()
                total_loss = total_loss + var_reg_loss
                losses["var_reg_loss"].append(var_reg_loss.item())

            if args.ratio_reg:
                ratio_reg_loss = ratio_reg(t, density_head, z, shift_type)
                ratio_reg_loss = ratio_reg_loss + (1 / n) * ratio_reg.prior()
                total_loss = total_loss + ratio_reg_loss
                losses["ratio_reg_loss"].append(ratio_reg_loss.item())

            if args.pos_reg:
                pos_reg_loss = pos_reg(t, model, x, shift_type)
                pos_reg_loss = pos_reg_loss + (1 / n) * pos_reg.prior()
                total_loss = total_loss + pos_reg_loss
                losses["pos_reg_loss"].append(pos_reg_loss.item())

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
                            t=t,
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
                s3 = f"pos_{int(args.pos_reg)}"
                s4 = f"dout_{args.dropout}"
                s5 = f"erm_{int(args.erm)}"
                exp_dir = f"{'-'.join([s1, s2, s3, s4, s5])}/{args.seed:02d}"

                # save estimated curve dataset
                results_path = f"{args.rdir}/{args.dataset}/{exp_dir}/curve.csv"
                os.makedirs(os.path.dirname(results_path), exist_ok=True)
                df.round(4).to_csv(results_path, index=False)

                # save experiment config
                config_path = f"{args.rdir}/{args.dataset}/{exp_dir}/config.yaml"
                with open(config_path, "w") as io:
                    yaml.safe_dump(vars(args), io)

                # save metrics
                metrics_path = f"{args.rdir}/{args.dataset}/{exp_dir}/metrics.yaml"
                metrics = {k: float(np.mean(v)) for k, v in losses.items()}
                metrics["last_saved_epoch"] = epoch
                metrics["@0.1"] = float(errors[0.1])
                metrics["@0.5"] = float(errors[0.5])
                metrics["curve"] = float(curve_error)
                metrics["seed"] = args.seed
                with open(metrics_path, "w") as io:
                    yaml.safe_dump(metrics, io)

                # plot curves
                _, ax = plt.subplots(1, 2, figsize=(6, 3))
                ax[0].plot(
                    df.delta, df.train_truth, label="train truth", c="blue", ls="-"
                )
                ax[0].plot(
                    df.delta, df.train_estim, label="train estim", c="blue", ls="--"
                )
                ax[1].plot(df.delta, df.test_truth, label="test truth", c="red", ls="-")
                ax[1].plot(
                    df.delta, df.test_estim, label="test estim", c="red", ls="--"
                )
                fig_path = f"{args.rdir}/{args.dataset}/{exp_dir}/fig.png"
                ax[0].legend()
                ax[1].legend()
                plt.savefig(fig_path)
                plt.close()
            model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--dataset", default="ihdp-N", type=str, choices=DATASETS)
    parser.add_argument("--rdir", default="results", type=str)
    parser.add_argument("--opt", default="sgd", type=str, choices=("adam", "sgd"))
    parser.add_argument("--n_train", default=500, type=int)
    parser.add_argument("--n_test", default=200, type=int)
    parser.add_argument("--n_epochs", default=10000, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--n_grid", default=10, type=int)
    parser.add_argument("--wd", default=5e-3, type=float)
    parser.add_argument("--silent", default=False, action="store_true")
    parser.add_argument("--dropout", default=0.0, type=float)

    # regularizations available
    parser.add_argument("--no_erm", default=True, dest="erm", action="store_false")
    parser.add_argument("--var_reg", default=False, action="store_true")
    parser.add_argument("--ratio_reg", default=False, action="store_true")
    parser.add_argument("--combo_reg", default=False, action="store_true")
    parser.add_argument("--pos_reg", default=False, action="store_true")
    parser.add_argument(
        "--no_fit_ratio_scale", dest="fit_ratio_scale", default=True, action="store_false"
    )
    parser.add_argument("--reg_multiscale", default=False, action="store_true")

    args = parser.parse_args()
    main(args)
