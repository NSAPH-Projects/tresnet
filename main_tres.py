import numpy as np
import torch
import torch.nn.functional as F
import argparse
from collections import defaultdict, deque
import pandas as pd
import os
import yaml
import matplotlib.pyplot as plt
from copy import deepcopy

import matplotlib
matplotlib.use('agg')  # needed in the cluster

from models.models import Drnet, VCNet, RatioNet
from utils import ratios
from models.modules import TargetedRegularizerCoeff
from dataset.dataset import get_iter, make_dataset, DATASETS, set_seed


def main(args: argparse.Namespace) -> None:
    # seed and use gpu when available
    set_seed(1234 + 131 * args.seed)  # like torch manual seed at all levels
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # experiment_dir
    s1 = f"var_{int(args.var_reg)}"
    s2 = f"ratio_{int(args.ratio_reg)}"
    s3 = f"pos_{int(args.pos_reg)}"
    s4 = f"dout_{args.dropout}"
    s5 = f"ratio_{args.ratio}"
    if args.edir is None:
        edir = f"{'-'.join([s1, s2, s3, s4, s5])}/{args.seed:02d}"
    else:
        edir = f"{args.edir}/{args.seed:02d}"
    os.makedirs(f"{args.rdir}/{args.dataset}/{edir}", exist_ok=True)

    # these are the shifting values use in the srf curve
    steps = 10
    delta_list = np.linspace(0.5 / steps, 0.5, num=steps, dtype=np.float32).tolist()

    # make dataset from available optionss
    D = make_dataset(args.dataset, delta_list, n_train=args.n_train, n_test=args.n_test, noise_scale=args.noise, count=args.count)
    train_matrix = D["train_matrix"].to(dev)
    test_matrix = D["train_matrix"].to(dev)
    shift_type = D["shift_type"]
    n, input_dim = train_matrix.shape[0], train_matrix.shape[1] - 2

    # make neural network model
    density_estimator_config = [(input_dim, 50, 1), (50, 50, 1)]
    pred_head_config = [(50, 50, 1), (50, 1, 1)]

    if args.ratio != "c_ratio" and not args.drnet:
        model = VCNet(
            density_estimator_config,
            num_grids=args.n_grid,
            pred_head_config=pred_head_config,
            spline_degree=2,
            spline_knots=[0.33, 0.66],
            dropout=args.dropout,
        ).to(dev)
        density_head = model.density_estimator

        if args.outcome_only:
            density_head.requires_grad_(False)
        if args.ratio_only:
            model.prediction_head.requires_grad_(False)
    elif args.ratio == "c_ratio" and not args.drnet:
        model = RatioNet(
            delta_list,
            density_estimator_config,
            num_grids=args.n_grid,
            pred_head_config=pred_head_config,
            spline_degree_Q=2,
            spline_knots_Q=[0.33, 0.66],
            spline_degree_W=2,
            spline_knots_W=np.linspace(min(delta_list), max(delta_list), num=len(delta_list)//2)[1:-1].tolist(),
            dropout=args.dropout,
        ).to(dev)
        ratio_head = model.ratio_estimator
        if args.outcome_only:
            ratio_head.requires_grad_(False)
        if args.ratio_only:
            model.prediction_head.requires_grad_(False)
    
    elif args.drnet:
        cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
        model = Drnet(
            density_estimator_config,
            isenhance=1,  # isenhance=0 is tarnet accorindg to vcnet repo
            num_grids=args.n_grid,
            pred_head_config=cfg,
            dropout=args.dropout,
        ).to(dev)
        # args.outcome_only = True
        density_head = model.density_estimator
        if args.outcome_only:
            density_head.requires_grad_(False)       



    model._initialize_weights()
    optim_params = [{"params": model.parameters(), "weight_decay": args.wd}]

    # set regularizations if other flags require it
    if args.combo_reg:
        args.var_reg = True
        args.ratio_reg = True

    # make regularizing layers if required
    if args.var_reg:
        var_reg = ratios.VarianceRegularizer(
            delta_list=delta_list, multiscale=args.reg_multiscale
        ).to(dev)
        optim_params.append({"params": var_reg.parameters()})

    if args.ratio_reg or args.ratio == "gps_ratio":
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

    if args.pos_reg_tr:
        pos_reg = ratios.PosteriorRegularizerTR(
            delta_list=delta_list,
            multiscale=args.reg_multiscale,
        ).to(dev)
        optim_params.append({"params": pos_reg.parameters()})

    if args.tr == "discrete" or not args.tr_reg:
        targeted_regularizer = torch.zeros(len(delta_list), device=dev)
    elif args.tr == "vc":
        targeted_regularizer = TargetedRegularizerCoeff(
            degree=2,
            knots=delta_list[::2],
        ).to(dev)
        tr_params = targeted_regularizer.parameters()
        optim_params.append(
            {"params": tr_params, "momentum": 0.0, "weight_decay": 0.0}
        )

    # make optimizer
    if args.opt == "adam":
        optimizer = torch.optim.Adam(optim_params, lr=3e-4)
    elif args.opt == "sgd":
        optimizer = torch.optim.SGD(
            optim_params, lr=args.lr, momentum=0.9, nesterov=True
        )

    best_loss, best_model, best_iter = 1e6, deepcopy(model), 0
    if args.tr_reg:
        best_tr = deepcopy(targeted_regularizer) if args.tr == "vc" else targeted_regularizer.clone()
    best_model.eval()

    # training loop
    train_loader = get_iter(train_matrix, batch_size=args.batch_size, shuffle=True)

    eps_lr = 0.25 / len(train_loader)

    for epoch in range(args.n_epochs):
        # dict to store all the losses per batch
        losses = defaultdict(lambda: deque(maxlen=len(train_loader)))

        # iterate each batch
        # for _, item in enumerate(train_loader):
        for _, (t, x, y) in enumerate(train_loader):
            total_loss = torch.tensor(0.0, device=dev)

            # move tensor to gpu if available
            # t = item["treatment"].to(dev)
            # x = item["covariates"].to(dev)
            # y = item["outcome"].to(dev)

            # zero grad and evaluate model
            optimizer.zero_grad()
            model_output = model(t, x)
            z = model_output["z"]

            # 1. density negative loglikelihood loss
            if args.ratio == "erm" and not args.outcome_only:
                probs = model_output["prob_score"]
                density_negll = -probs.log().mean()
                losses["density_negll"].append(density_negll.item())
                if not args.outcome_only:
                    total_loss = total_loss + density_negll

            elif args.ratio == "gps_ratio" and not args.outcome_only:
                gps_ratio_loss = ratio_reg(t, density_head, z, shift_type)
                gps_ratio_loss = gps_ratio_loss + (1 / n) * ratio_reg.prior()
                if not args.outcome_only:
                    total_loss = total_loss + gps_ratio_loss
                losses["gps_ratio_loss"].append(gps_ratio_loss.item())

            elif args.ratio == "c_ratio" and not args.outcome_only:
                gps_ratio_losses = []
                for j, d in enumerate(delta_list):
                    t_d = ratios.shift(t, d, shift_type)
                    logits = torch.cat([model.log_ratio(t_d, z, j), model.log_ratio(t, z, j)])
                    tgts = torch.cat([torch.ones_like(t), torch.zeros_like(t)]).clamp(args.ls, 1 - args.ls)
                    L = F.binary_cross_entropy_with_logits(logits, tgts)
                    gps_ratio_losses.append(L)
                gps_ratio_loss = sum(gps_ratio_losses) / len(delta_list)
                if not args.outcome_only:
                    total_loss = total_loss + gps_ratio_loss
                losses["gps_ratio_loss"].append(gps_ratio_loss.item())

            # 2. outcome loss
            y_hat = model_output["predicted_outcome"]

            if not args.poisson:
                outcome_loss = F.mse_loss(y_hat, y)
            else:
                y_hat = y_hat.clamp(-10, 10)
                outcome_loss = F.poisson_nll_loss(y_hat, y, log_input=True)

            losses["outcome_loss"].append(outcome_loss.item())
            if not args.ratio_only:
                total_loss = total_loss + outcome_loss

            # 3. targeted loss
            # make perturbed predictor
            tr_losses = []
            biases = torch.zeros(len(delta_list), device=dev)
            for j, d in enumerate(delta_list):
                if args.ratio != "c_ratio":
                    log_ratio = ratios.log_density_ratio_under_shift(
                        t=t,
                        delta=torch.full_like(t,d),
                        density_estimator=density_head,
                        z=z,
                        shift_type=shift_type,
                    )
                else:
                    log_ratio = model.log_ratio(t, z, j)
                if args.tr == "discrete":
                    eps = targeted_regularizer[j]
                elif args.tr == "vc":
                    eps = targeted_regularizer(torch.full_like(t, d))
                ratio = log_ratio.clamp(-10, 10).exp()
                ratio_ = ratio.detach() if args.detach_ratio else ratio
                if args.ratio_norm:
                    ratio_ = ratio_ / ratio_.mean()
                if args.pert == "simple":
                    y_pert = y_hat + eps
                    if not args.poisson:
                        L = (ratio_ * (y_pert - y).pow(2)).mean()
                        with torch.no_grad():
                            bias = (ratio_ * (y - y_hat)).mean() / ratio_.mean()
                    else:
                        y_pert = y_pert.clamp(-10, 10)
                        L = (ratio_ * F.poisson_nll_loss(y_pert, y, log_input=True, reduction='none')).mean()
                        with torch.no_grad():
                            bias = (ratio_ * y + 1e-8).mean().log() - (ratio_ * y_hat.exp() + 1e-8).mean().log()
                            # bias = bias.clamp(-0.05, 0.05)
                elif args.pert == "original":
                    y_pert = y_hat + ratio_ * eps
                    if not args.poisson:
                        L = F.mse_loss(y_pert, y)
                        with torch.no_grad():
                            bias = (ratio_ * (y - y_hat)).mean() / ratio_.pow(2).mean()
                    else:
                        raise NotImplementedError("poisson can  only be done with simple pert")
                if args.tr == "discrete" and args.pert == "simple": # manual update of epsilon
                    biases[j] = bias
                    # eps.add_((bias - eps).item(), alpha=args.eps_lr)
                tr_losses.append(L)
            tr_losses = sum(tr_losses)
            if args.tr_reg:
                total_loss = total_loss + args.beta * tr_losses
            losses["tr_loss"].append(tr_losses.item())


            # ix = torch.randint(0, len(delta_list), size=(t.shape[0],), device=dev)
            # random_delta = torch.FloatTensor(delta_list).to(dev)[ix]
            # if args.tr == "discrete":
            #     eps = targeted_regularizer[ix]
            # elif args.tr == "vc":
            #     eps = targeted_regularizer(random_delta)
            # ratio = ratios.log_density_ratio_under_shift(
            #     t=t,
            #     delta=random_delta,
            #     density_estimator=density_head,
            #     z=z,
            #     shift_type=shift_type,
            # )
            # ratio = ratio.clamp(-10, 10).exp()
            # ratio_ = ratio.detach() if args.detach_ratio else ratio
            # if args.pert == "original":
            #     y_pert = y_hat + eps * ratio_
            #     tr_loss = F.mse_loss(y_pert, y)
            # elif args.pert == "simple":
            #     y_pert = y_hat + eps
            #     if args.ratio_norm:
            #         ratio_ = ratio_ / ratio_.mean()
            #     tr_loss = (ratio_ * (y_pert - y).pow(2)).mean()
            # 
            # losses["tr_loss"].append(tr_loss.item())
            # if args.tr_reg:
            #     total_loss = total_loss + args.beta * tr_loss

            # 4. other regularization losses
            if args.var_reg:
                var_reg_loss = var_reg(t, density_head, z, shift_type)
                var_reg_loss = var_reg_loss + (1 / n) * var_reg.prior()
                total_loss = total_loss + var_reg_loss
                losses["var_reg_loss"].append(var_reg_loss.item())

            if args.pos_reg:
                pos_reg_loss = pos_reg(t, model, x, shift_type)
                pos_reg_loss = pos_reg_loss + (1 / n) * pos_reg.prior()
                total_loss = total_loss + pos_reg_loss
                losses["pos_reg_loss"].append(pos_reg_loss.item())

            losses["total_loss"].append(total_loss.item())

            total_loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 10.0)
            optimizer.step()

            if args.tr == 'discrete':
                targeted_regularizer += eps_lr * (biases - targeted_regularizer)
            # update epsilon (coordinate ascent)

        # evaluation
        if (epoch + 1) % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                # replace best model if improves
                if args.val in ("is", "test"):
                    M = test_matrix
                    t, x, y = M[:, 0], M[:, 1:-1], M[:, -1]
                    output = model.forward(t, x)
                    z = output["z"]
                    y_hat = output["predicted_outcome"]
                    val_losses = []
                    for j, d in enumerate(delta_list):
                        if args.val == "test":
                            ratio = torch.ones_like(t)
                        else:
                            if args.ratio != "c_ratio":
                                log_ratio = ratios.log_density_ratio_under_shift(
                                    t=t,
                                    delta=torch.full_like(t, d),
                                    density_estimator=best_model.density_estimator,
                                    z=z,
                                    shift_type=D["shift_type"],
                                )
                            else:
                                log_ratio = model.log_ratio(t, z, j)
                            ratio = log_ratio.clamp(-10, 10).exp()
                        if args.ratio_norm:
                            ratio = ratio / ratio.mean()
                        # ratio = 1
                        if args.tr_reg:
                            if args.tr == "discrete":
                                eps = best_tr[j]
                            elif args.tr == "vc":
                                eps = best_tr(torch.full_like(t, d))
                            if args.pert == "original":
                                y_pert = y_hat + eps * ratio
                            elif args.pert == "simple":
                                y_pert= y_hat + eps
                        else:
                            y_pert = y_hat
                        if args.poisson:
                            y_hat = y_hat.clamp(-10, 10)
                            y_pert = y_pert.clamp(-10, 10)
                            L = (ratio * F.poisson_nll_loss(y_pert, y, reduction='none')).mean()
                            val_losses.append(L.item())
                        else:
                            val_losses.append((ratio * (y_pert - y).pow(2)).mean().item())
                    val_loss = float(np.mean(val_losses))
                    if val_loss < best_loss:
                        best_model = deepcopy(model)
                        best_model.eval()
                        best_loss = val_loss
                        best_iter = epoch
                        best_tr = deepcopy(targeted_regularizer) if args.tr == "vc" else targeted_regularizer.clone()

                elif args.val == "none":
                    best_model = deepcopy(model)
                    best_model.eval()

                # obtain all evaluation metrics

                if not args.silent:
                    print("== Epoch: ", epoch, " ==")
                    print("Metrics:")
                    for k, vec in losses.items():
                        print(f"  {k}: {np.mean(vec):.4f}")

                # iptw estimates
                df = pd.DataFrame({"delta": delta_list})
                for part in ("train", "test"):
                    M = train_matrix if part == "train" else test_matrix
                    t, x, y = M[:, 0], M[:, 1:-1], M[:, -1]
                    z = best_model.forward(t, x)["z"]
                    srf = D["srf_" + part]
                    df[part + "_truth"] = srf

                    # dictionaries for all kind of estimates
                    ipw_estims = []
                    ipw_errors = []
                    aipw_estims = []
                    aipw_errors = []
                    # tmle_estims = []
                    # tmle_errors = []
                    tr_estims = []
                    tr_errors = []
                    plugin_estims = []
                    plugin_errors = []

                    shift_type = D["shift_type"]

                    for j, (d, truth) in enumerate(zip(delta_list, srf)):
                        if args.ratio != "c_ratio":
                            log_ratio = ratios.log_density_ratio_under_shift(
                                t=t,
                                delta=torch.full_like(t, d),
                                density_estimator=density_head,
                                z=z,
                                shift_type=shift_type,
                            )
                        else:
                            log_ratio = model.log_ratio(t, z, j)
                        ratio = log_ratio.clamp(-10, 10).exp()
                        if args.ratio_norm:
                            ratio = ratio / ratio.mean()
                        estim = (ratio * y).mean().item()
                        error = (estim - truth).item()
                        ipw_estims.append(estim)
                        ipw_errors.append(error)

                        # A-IPTW estimates
                        t_delta = ratios.shift(t, d, shift_type)
                        y_delta = best_model(t_delta, x)["predicted_outcome"]
                        if args.tr == "discrete":
                            eps = best_tr[j]
                        elif args.tr == "vc":
                            eps = best_tr(torch.full_like(t, d))
                        if args.pert == "original":
                            y_pert = y_hat + eps * ratio
                            y_pert_delta = y_delta + eps * ratio
                        elif args.pert == "simple":
                            y_pert = y_hat + eps
                            y_pert_delta = y_delta + eps
                        # y_hat = best_model(t, x)["predicted_outcome"]
                        if args.poisson:
                            y_delta = y_delta.clamp(-10, 10)
                            y_pert = torch.exp(y_pert)
                            y_pert_delta = torch.exp(y_pert_delta)
                            y_delta = torch.exp(y_delta)
                        estim = (ratio * (y - y_pert)).mean() + y_pert_delta.mean()
                        error = (estim - truth)
                        aipw_estims.append(estim.item())
                        aipw_errors.append(error.item())

                        # Targeted Regularization
                        estim = y_pert_delta.mean()
                        error = (estim - truth)
                        tr_estims.append(estim.item())
                        tr_errors.append(error.item())

                        # Plugin
                        estim = y_delta.mean()
                        error = (estim - truth)
                        plugin_estims.append(estim.item())
                        plugin_errors.append(error.item())

                    # add estimation error as columns of result dataframe
                    df[part + "_ipw_estim"] = ipw_estims
                    df[part + "_ipw_error"] = ipw_errors
                    df[part + "_aipw_estim"] = aipw_estims
                    df[part + "_aipw_error"] = aipw_errors
                    df[part + "_tr_estim"] = tr_estims
                    df[part + "_tr_error"] = tr_errors
                    df[part + "_plugin_estim"] = plugin_estims
                    df[part + "_plugin_error"] = plugin_errors

                    # save metrics #TODO: this is only doing test, must upate to use df
                    # for computation
                    # TODO, separate test and train metrics
                    metrics = {k: float(np.mean(v)) for k, v in losses.items()}
                    # metrics["ipw_curve_error"] = float(
                    #     np.square(ipw_errors).mean() ** 0.5
                    # )
                    metrics["ipw_curve_rmse"] = float(
                        np.square(ipw_errors).mean() ** 0.5
                    )
                    metrics["aipw_curve_rmse"] = float(
                        np.square(aipw_errors).mean() ** 0.5
                    )
                    metrics["tr_curve_rmse"] = float(
                        np.square(tr_errors).mean() ** 0.5
                    )
                    metrics["plugin_curve_rmse"] = float(
                        np.square(plugin_errors).mean() ** 0.5
                    )
                    metrics["ipw_bias"] = float(np.mean(ipw_errors))
                    metrics["aipw_curve_bias"] = float(np.mean(aipw_errors))
                    metrics["tr_curve_bias"] = float(np.mean(aipw_errors))
                    metrics["pluglin_curve_bias"] = float(np.mean(aipw_errors))
                    metrics_path = (
                        f"{args.rdir}/{args.dataset}/{edir}/metrics_{part}.yaml"
                    )
                    metrics["last_saved_epoch"] = epoch
                    metrics["best_iter"] = best_iter
                    metrics["best_val"] = best_loss
                    with open(metrics_path, "w") as io:
                        yaml.safe_dump(metrics, io)

                    if not args.silent:
                        print(f"SRF {part}:")
                        # print(f"  ipw curve: {metrics['ipw_curve_error']:.4f}")
                        print(f"  aipw curve: {metrics['aipw_curve_rmse']:.4f}")
                        print(f"  tr curve: {metrics['tr_curve_rmse']:.4f}")


                # save estimated curve dataset
                results_path = f"{args.rdir}/{args.dataset}/{edir}/curve.csv"
                df.round(4).to_csv(results_path, index=False)

                # save experiment config
                config_path = f"{args.rdir}/{args.dataset}/{edir}/config.yaml"
                with open(config_path, "w") as io:
                    yaml.safe_dump(vars(args), io)

                # plot curves
                _, ax = plt.subplots(1, 2, figsize=(6, 3))
                ax[0].plot(
                    df.delta, df.train_truth, label="truth (train)", c="black", ls="--"
                )
                ax[0].plot(
                    df.delta, df.train_tr_estim, label="tresnet", 
                )
                ax[0].plot(
                    df.delta, df.train_aipw_estim, label="aipw",
                )
                ax[1].plot(df.delta, df.test_truth, label="truth (test)", c="black", ls="--")
                ax[1].plot(df.delta, df.test_tr_estim, label="tresnet")
                ax[1].plot(df.delta, df.test_aipw_estim, label="aipw")
                fig_path = f"{args.rdir}/{args.dataset}/{edir}/fig.png"
                ax[0].legend()
                ax[1].legend()
                plt.savefig(fig_path)
                plt.close()
            model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--n_grid", default=30, type=int)
    parser.add_argument("--dataset", default="news-N", type=str, choices=DATASETS)
    parser.add_argument(
        "--pert", default="simple", type=str, choices=("original", "simple")
    )
    parser.add_argument("--detach_ratio", default=False, action="store_true")
    parser.add_argument("--rdir", default="results", type=str)
    parser.add_argument("--edir", default=None, type=str)
    parser.add_argument("--opt", default="sgd", type=str, choices=("adam", "sgd"))
    parser.add_argument("--val", default="test", type=str, choices=("is", "val", "none"))
    parser.add_argument("--n_train", default=500, type=int)
    parser.add_argument("--n_test", default=200, type=int)
    parser.add_argument("--n_epochs", default=5000, type=int)
    parser.add_argument("--batch_size", default=4000, type=int)
    parser.add_argument("--eval_every", default=100, type=int)
    parser.add_argument("--wd", default=5e-3, type=float)
    parser.add_argument("--lr", default=1e-4, type=float) 
    parser.add_argument("--ls", default=0.1, type=float) 
    # parser.add_argument("--eps_lr", default=0.1, type=float) 
    parser.add_argument("--beta", default=0.1, type=float)
    parser.add_argument("--noise", default=0.5, type=float)  # it is 0.5 in vcnet paper
    parser.add_argument("--silent", default=False, action="store_true")
    parser.add_argument("--ratio_norm", default=True, action="store_true")
    parser.add_argument("--dropout", default=0.05, type=float)

    # regularizations availables
    parser.add_argument("--ratio", default="gps_ratio", type=str, choices=("erm", "gps_ratio", "c_ratio"))
    parser.add_argument("--var_reg", default=False, action="store_true")
    parser.add_argument("--outcome_only", default=False, action="store_true")
    parser.add_argument("--ratio_only", default=False, action="store_true")
    parser.add_argument("--ratio_reg", default=False, action="store_true")
    parser.add_argument("--combo_reg", default=False, action="store_true")
    parser.add_argument("--pos_reg", default=False, action="store_true")
    parser.add_argument("--pos_reg_tr", default=False, action="store_true")
    parser.add_argument("--tr_reg", default=False, action="store_true")
    parser.add_argument("--poisson", default=False, action="store_true")
    parser.add_argument("--count", default=False, action="store_true")
    parser.add_argument("--drnet", default=False, action="store_true")
    parser.add_argument("--target", type=str, default="si", choices=("si", "erf"))
    parser.add_argument("--tr", default="discrete", choices=("discrete", "vc"))
    parser.add_argument("--fit_ratio_scale", default=False, action="store_true")
    parser.add_argument("--reg_multiscale", default=False, action="store_true")

    args = parser.parse_args()

    with torch.autograd.set_detect_anomaly(True):
        main(args)
