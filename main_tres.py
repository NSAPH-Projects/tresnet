import numpy as np
import torch
import torch.nn.functional as F
import argparse
from collections import defaultdict, deque
import pandas as pd
import os
import yaml
import matplotlib.pyplot as plt

plt.ioff()

from models.VCNet import VCNet
from utils import ratios
from models.modules import TargetedRegularizerCoeff
from dataset.dataset import get_iter, make_dataset, DATASETS, set_seed


def main(args: argparse.Namespace) -> None:
    # seed and use gpu when available
    set_seed(1234 + 131 * args.seed)  # like torch manual seed at all levels
    dev = torch.cuda if torch.cuda.is_available() else "cpu"

    # experiment_dir
    s1 = f"var_{int(args.var_reg)}"
    s2 = f"ratio_{int(args.ratio_reg)}"
    s3 = f"pos_{int(args.pos_reg)}"
    s4 = f"dout_{args.dropout}"
    s5 = f"erm_{int(args.erm)}"
    exp_dir = f"{'-'.join([s1, s2, s3, s4, s5])}/{args.seed:02d}"
    os.makedirs(f"{args.rdir}/{args.dataset}/{exp_dir}", exist_ok=True)

    # these are the shifting values use in the srf curve
    steps = 10
    delta_list = torch.linspace(0.5 / steps, 0.5, steps=steps).to(dev)

    # make dataset from available options
    D = make_dataset(args.dataset, delta_list, n_train=args.n_train, n_test=args.n_test)
    train_matrix = D["train_matrix"].to(dev)
    shift_type = D["shift_type"]
    n, input_dim = train_matrix.shape[0], train_matrix.shape[1] - 2

    # make neural network model
    density_estimator_config = [(input_dim, 50, 1), (50, 50, 1)]
    pred_head_config = [(50, 50, 1), (50, 1, 1)]

    model = VCNet(
        density_estimator_config,
        num_grids=args.n_grid,
        pred_head_config=pred_head_config,
        spline_degree=2,
        spline_knots=[0.33, 0.66],
        dropout=args.dropout,
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
        ).to(dev)
        optim_params.append({"params": var_reg.parameters()})

    if args.ratio_reg:
        ratio_reg = ratios.RatioRegularizer(
            delta_list=delta_list,
            multiscale=args.reg_multiscale,
            fit_scale=args.fit_ratio_scale,
        ).to(dev)
        optim_params.append({"params": ratio_reg.parameters()})

    if args.pos_reg:
        pos_reg = ratios.PosteriorRegularizer(
            delta_list=delta_list,
            multiscale=args.reg_multiscale,
        ).to(dev)
        optim_params.append({"params": pos_reg.parameters()})

    if args.tr == "discrete":
        targeted_regularizer = torch.zeros_like(delta_list)
        if args.tr_reg:
            targeted_regularizer.requires_grad_(True)
            optim_params.append({"params": targeted_regularizer, "lr": 1e-5, "weight_decay": 0.0})
    elif args.tr == "vc":
        targeted_regularizer_model = TargetedRegularizerCoeff(
            degree=2,
            knots=delta_list[::2],
        ).to(dev)
        if args.tr_reg:
            optim_params.append(
                {"params": targeted_regularizer_model.parameters(), "lr": 1e-5, "weight_decay": 0.0}
            )
        else:
            for p in targeted_regularizer_model.parameters():
                p.requires_grad_(False)

    # make optimizer
    if args.opt == "adam":
        optimizer = torch.optim.Adam(optim_params, lr=3e-4)
    elif args.opt == "sgd":
        optimizer = torch.optim.SGD(
            optim_params, lr=3e-5, momentum=0.9, nesterov=True
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
            model_output = model(t, x)
            probs = model_output["prob_score"]
            z = model_output["z"]

            # 1. density negative loglikelihood loss
            density_negll = -probs.log().mean()
            losses["density_negll"].append(density_negll.item())
            if args.erm:
                total_loss = total_loss + density_negll

            # 2. outcome loss
            y_hat = model_output["predicted_outcome"]
            outcome_loss = F.mse_loss(y_hat, y)
            losses["outcome_loss"].append(outcome_loss.item())
            total_loss = total_loss + outcome_loss

            # 3. targeted loss
            # make perturbed predictor for random delta
            ix = torch.randint(0, len(delta_list), size=(t.shape[0],), device=dev)
            random_delta = delta_list[ix]
            if args.tr == "discrete":
                eps = targeted_regularizer[ix]
            elif args.tr == "vc":
                eps = targeted_regularizer_model(random_delta)
            ratio = ratios.log_density_ratio_under_shift(
                treatment=t,
                delta=random_delta,
                density_estimator=density_head,
                z=z,
                shift_type=shift_type,
            )
            ratio = ratio.clamp(-10, 10).exp()
            ratio_ = ratio.detach() if args.detach_ratio else ratio
            if args.pert == "original":
                y_pert = y_hat + eps * ratio_
                tr_loss = F.mse_loss(y_pert, y)
            elif args.pert == "simple":
                y_pert = y_hat + eps
                if args.ratio_normalize:
                    ratio_ = ratio_ / ratio_.sum()
                tr_loss = (ratio_ * (y_pert - y).pow(2)).mean()
            losses["tr_loss"].append(tr_loss.item())
            if args.tr_reg:
                total_loss = total_loss + tr_loss

            # 4. other regularization losses
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

                    # dictionaries for all kind of estimates
                    # ipw_estims = []
                    # ipw_errors = []
                    aipw_estims = []
                    aipw_errors = []
                    # tmle_estims = []
                    # tmle_errors = []
                    tr_estims = []
                    tr_errors = []

                    shift_type = D["shift_type"]

                    for j, (d, truth) in enumerate(zip(delta_list, srf)):
                        # IPW estimates
                        # obtain importance sampling density ratio weights
                        log_ratio = ratios.log_density_ratio_under_shift(
                            treatment=t,
                            delta=d,
                            density_estimator=density_head,
                            z=z,
                            shift_type=shift_type,
                        )
                        ratio = log_ratio.clamp(-10, 10).exp()
                        if args.ratio_normalize:
                            ratio = ratio / ratio.sum()
                        estim = (ratio * y).sum().item()
                        error = (estim - truth).item()
                        # ipw_estims.append(estim)
                        # ipw_errors.append(error)

                        # A-IPTW estimates
                        t_delta = ratios.shift(t, d, shift_type)
                        y_delta = model(t_delta, x)["predicted_outcome"]
                        y_hat = model(t, x)["predicted_outcome"]
                        estim = (ratio * (y - y_hat)).sum() + y_delta.mean()
                        error = (estim - truth).item()
                        aipw_estims.append(estim)
                        aipw_errors.append(error)

                        # Targeted Regularization
                        if args.tr == "discrete":
                            eps = targeted_regularizer[j]
                        elif args.tr == "vc":
                            eps = targeted_regularizer_model(torch.full_like(t, d))
                        if args.pert == "original":
                            y_pert_delta = y_delta + eps * ratio
                        elif args.pert == "simple":
                            y_pert_delta = y_delta + eps
                        estim = y_pert_delta.mean()
                        error = (estim - truth).item()
                        tr_estims.append(estim)
                        tr_errors.append(error)

                    # add estimation error as columns of result dataframe
                    # df[part + "_ipw_estim"] = ipw_estims
                    # df[part + "_ipw_error"] = ipw_errors
                    df[part + "_aipw_estim"] = aipw_estims
                    df[part + "_aipw_error"] = aipw_errors
                    df[part + "_tr_estim"] = tr_estims
                    df[part + "_tr_error"] = tr_estims

                    # save metrics #TODO: this is only doing test, must upate to use df
                    # for computation
                    # TODO, separate test and train metrics
                    metrics = {k: float(np.mean(v)) for k, v in losses.items()}
                    # metrics["ipw_curve_error"] = float(
                    #     np.square(ipw_errors).mean() ** 0.5
                    # )
                    metrics["aipw_curve_error"] = float(
                        np.square(aipw_errors).mean() ** 0.5
                    )
                    metrics["tr_curve_error"] = float(
                        np.square(tr_errors).mean() ** 0.5
                    )
                    metrics_path = (
                        f"{args.rdir}/{args.dataset}/{exp_dir}/metrics_{part}.yaml"
                    )
                    metrics["last_saved_epoch"] = epoch
                    with open(metrics_path, "w") as io:
                        yaml.safe_dump(metrics, io)

                    if not args.silent:
                        print(f"SRF {part}:")
                        # print(f"  ipw curve: {metrics['ipw_curve_error']:.4f}")
                        print(f"  aipw curve: {metrics['aipw_curve_error']:.4f}")
                        print(f"  tr curve: {metrics['tr_curve_error']:.4f}")

                # save estimated curve dataset
                results_path = f"{args.rdir}/{args.dataset}/{exp_dir}/curve.csv"
                df.round(4).to_csv(results_path, index=False)

                # save experiment config
                config_path = f"{args.rdir}/{args.dataset}/{exp_dir}/config.yaml"
                with open(config_path, "w") as io:
                    yaml.safe_dump(vars(args), io)

                # plot curves
                _, ax = plt.subplots(1, 2, figsize=(6, 3))
                ax[0].plot(
                    df.delta, df.train_truth, label="train truth", c="blue", ls="-"
                )
                ax[0].plot(
                    df.delta, df.train_tr_estim, label="train estim", c="blue", ls="--"
                )
                ax[1].plot(df.delta, df.test_truth, label="test truth", c="red", ls="-")
                ax[1].plot(
                    df.delta, df.test_tr_estim, label="test estim", c="red", ls="--"
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
    parser.add_argument("--n_grid", default=10, type=int)
    parser.add_argument("--dataset", default="ihdp-N", type=str, choices=DATASETS)
    parser.add_argument(
        "--pert", default="simple", type=str, choices=("original", "simple")
    )
    parser.add_argument("--detach_ratio", default=False, action="store_true")
    parser.add_argument("--rdir", default="results", type=str)
    parser.add_argument("--opt", default="sgd", type=str, choices=("adam", "sgd"))
    parser.add_argument("--n_train", default=500, type=int)
    parser.add_argument("--n_test", default=200, type=int)
    parser.add_argument("--n_epochs", default=5000, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--wd", default=5e-3, type=float)
    parser.add_argument("--silent", default=False, action="store_true")
    parser.add_argument("--ratio_normalize", default=False, action="store_true")
    parser.add_argument("--dropout", default=0.0, type=float)

    # regularizations available
    parser.add_argument("--no_erm", default=True, dest="erm", action="store_false")
    parser.add_argument("--var_reg", default=False, action="store_true")
    parser.add_argument("--ratio_reg", default=False, action="store_true")
    parser.add_argument("--combo_reg", default=False, action="store_true")
    parser.add_argument("--pos_reg", default=False, action="store_true")
    parser.add_argument("--tr_reg", default=False, action="store_true")
    parser.add_argument("--tr", default="discrete", choices=("discrete", "vc"))
    parser.add_argument(
        "--no_fit_ratio_scale",
        dest="fit_ratio_scale",
        default=True,
        action="store_false",
    )
    parser.add_argument("--reg_multiscale", default=False, action="store_true")

    args = parser.parse_args()
    main(args)
