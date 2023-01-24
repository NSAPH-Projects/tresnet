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

plt.ioff()

from models.VCNet import VCNet
from utils import ratios
from models.modules import TargetedRegularizerCoeff
#from dataset.dataset import get_iter, make_dataset, DATASETS, set_seed

from data_medicare.dataset_medicare import set_seed, get_iter, DataMedicare


def main(args: argparse.Namespace) -> None:
    # seed and use gpu when available
    set_seed(1234 + 131 * args.seed)  # like torch manual seed at all levels
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # experiment_dir
    s1 = f"var_{int(args.var_reg)}"
    s2 = f"ratio_{int(args.ratio_reg)}"
    s3 = f"pos_{int(args.pos_reg)}"
    s4 = f"dout_{args.dropout}"
    s5 = f"erm_{int(args.erm)}"
    if args.edir is None:
        edir = f"{'-'.join([s1, s2, s3, s4, s5])}/{args.seed:02d}"
    else:
        edir = f"{args.edir}/{args.seed:02d}"
    os.makedirs(f"{args.rdir}/{args.dataset}/{edir}", exist_ok=True)

    # these are the shifting values use in the srf curve
    steps = 10
    delta_list = np.linspace(0.5 / steps, 0.5, num=steps, dtype=np.float32).tolist()

    # make dataset from available optionss
    path = "./zip_data.csv"
    treatment_col= "pm25"
    outcome_col= "dead"
    offset_col= "time_count"
    categorical_variables = ["year", "regionNORTHEST", "regionSOUTH", "regionWEST"]
    columns_to_omit= ["zip"]

    data = DataMedicare(
                path, 
                treatment_col, 
                outcome_col, 
                offset_col, 
                categorical_variables=categorical_variables, 
                columns_to_omit=columns_to_omit, 
                train_prop=args.train_prop # defaults to 0.8, i.e 80% of data for training
            )
    data.init() # you have to init the DataMedicare object

    # data.training_set: contains training set in format dict[keys]
    # data.test: contains test set in format dict[keys]
    # data.treatment_norm_min and data.treatment_norm_max are the coefficients for transforming from (0, 1) 

    train_data = data.training_set
    test_data = data.test_set      
    
    for item in train_data:
        train_data[item] = train_data[item].to(dev)
    
    for item in train_data:
        test_data[item] = test_data[item].to(dev)
        
    n, input_dim = train_data["covariates"].shape[0], train_data["covariates"].shape[1] 

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
        args.ratio_reg = True
        args.fit_ratio_scale = False

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

    if args.pos_reg_tr:
        pos_reg = ratios.PosteriorRegularizerTR(
            delta_list=delta_list,
            multiscale=args.reg_multiscale,
        ).to(dev)
        optim_params.append({"params": pos_reg.parameters()})

    if args.tr == "discrete":
        targeted_regularizer = torch.zeros(len(delta_list), requires_grad=args.tr_reg, device=dev)
        tr_params = targeted_regularizer
    elif args.tr == "vc":
        targeted_regularizer = TargetedRegularizerCoeff(
            degree=2,
            knots=delta_list[::2],
        ).to(dev)
        tr_params = targeted_regularizer.parameters()

    if args.tr_reg:
        optim_params.append(
            {"params": tr_params, "lr": args.lr, "momentum": 0.0, "weight_decay": 0.0}
        )
    else:
        if args.tr == "vc":
            for p in tr_params:
                p.requires_grad_(False)

    # make optimizer
    if args.opt == "adam":
        optimizer = torch.optim.Adam(optim_params, lr=3e-4)
    elif args.opt == "sgd":
        optimizer = torch.optim.SGD(
            optim_params, lr=args.lr, momentum=0.9, nesterov=True
        )

    best_loss, best_model, best_iter = 1e6, deepcopy(model), 0
    if args.tr_reg:
        best_tr = deepcopy(targeted_regularizer)
    best_model.eval()

    # training loop
    train_loader = get_iter(train_data, batch_size=args.batch_size)

    for epoch in range(args.n_epochs):
        # dict to store all the losses per batch
        losses = defaultdict(lambda: deque(maxlen=len(train_loader)))

        # iterate each batch
        # for _, item in enumerate(train_loader):
        for _, (t, x, y, offset) in enumerate(train_loader):

            # Now the offset is avilable as above, try it out 

            total_loss = torch.tensor(0.0, device=dev)

            # move tensor to gpu if available
            # t = item["treatment"].to(dev)
            # x = item["covariates"].to(dev)
            # y = item["outcome"].to(dev)

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
            random_delta = torch.FloatTensor(delta_list).to(dev)[ix]
            if args.tr == "discrete":
                eps = targeted_regularizer[ix]
            elif args.tr == "vc":
                eps = targeted_regularizer(random_delta)
            ratio = ratios.log_density_ratio_under_shift(
                t=t,
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
                    ratio_ = ratio_ / ratio_.mean()
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
        if epoch == 0 or (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                # replace best model if improves
                if args.val == "is":
                    M = test_data
                    t, x, y = M["treatment"], M["covariates"], M["outcome"]
                    output = model.forward(t, x)
                    z, y_hat = output["z"], output["predicted_outcome"]
                    val_losses = []
                    for j, d in enumerate(delta_list):
                        log_ratio = ratios.log_density_ratio_under_shift(
                            t=t,
                            delta=torch.full_like(t, d),
                            density_estimator=best_model.density_estimator,
                            z=z,
                            shift_type=D["shift_type"],
                        )
                        ratio = log_ratio.clamp(-10, 10).exp()
                        if args.ratio_normalize:
                            ratio = ratio / ratio.mean()
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
                        val_losses.append((ratio * (y_pert - y).pow(2)).mean().item())
                    val_loss = float(np.mean(val_losses))
                    if val_loss < best_loss:
                        best_model = deepcopy(model)
                        best_model.eval()
                        best_loss = val_loss
                        best_iter = epoch
                        best_tr = deepcopy(targeted_regularizer)

                elif args.val is None:
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
                    M = train_data if part == "train" else test_data
                    t, x, y = M["treatment"], M["covariates"], M["outcome"]
                    z = best_model.forward(t, x)["z"]
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
                    plugin_estims = []
                    plugin_errors = []

                    shift_type = D["shift_type"]

                    for j, (d, truth) in enumerate(zip(delta_list, srf)):
                        # IPW estimates
                        # obtain importance sampling density ratio weights
                        log_ratio = ratios.log_density_ratio_under_shift(
                            t=t,
                            delta=torch.full_like(t, d),
                            density_estimator=density_head,
                            z=z,
                            shift_type=shift_type,
                        )
                        ratio = log_ratio.clamp(-10, 10).exp()
                        if args.ratio_normalize:
                            ratio = ratio / ratio.mean()
                        estim = (ratio * y).mean().item()
                        error = (estim - truth).item()
                        # ipw_estims.append(estim)
                        # ipw_errors.append(error)

                        # A-IPTW estimates
                        t_delta = ratios.shift(t, d, shift_type)
                        y_delta = best_model(t_delta, x)["predicted_outcome"]
                        y_hat = best_model(t, x)["predicted_outcome"]
                        estim = (ratio * (y - y_hat)).mean() + y_delta.mean()
                        error = (estim - truth)
                        aipw_estims.append(estim.item())
                        aipw_errors.append(error.item())

                        # Targeted Regularization
                        if args.tr == "discrete":
                            eps = best_tr[j]
                        elif args.tr == "vc":
                            eps = best_tr(torch.full_like(t, d))
                        if args.pert == "original":
                            y_pert_delta = y_delta + eps * ratio
                        elif args.pert == "simple":
                            y_pert_delta = y_delta + eps
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
                    # df[part + "_ipw_estim"] = ipw_estims
                    # df[part + "_ipw_error"] = ipw_errors
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
                    metrics["aipw_curve_error"] = float(
                        np.square(aipw_errors).mean() ** 0.5
                    )
                    metrics["tr_curve_error"] = float(
                        np.square(tr_errors).mean() ** 0.5
                    )
                    metrics["plugin_curve_error"] = float(
                        np.square(plugin_errors).mean() ** 0.5
                    )
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
                        print(f"  aipw curve: {metrics['aipw_curve_error']:.4f}")
                        print(f"  tr curve: {metrics['tr_curve_error']:.4f}")


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
                    df.delta, df.train_truth, label="train truth", c="blue", ls="-"
                )
                ax[0].plot(
                    df.delta, df.train_tr_estim, label="train estim", c="blue", ls="--"
                )
                ax[1].plot(df.delta, df.test_truth, label="test truth", c="red", ls="-")
                ax[1].plot(
                    df.delta, df.test_tr_estim, label="test estim", c="red", ls="--"
                )
                fig_path = f"{args.rdir}/{args.dataset}/{edir}/fig.png"
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
        "--pert", default="original", type=str, choices=("original", "simple")
    )
    parser.add_argument("--detach_ratio", default=False, action="store_true")
    parser.add_argument("--rdir", default="results", type=str)
    parser.add_argument("--edir", default=None, type=str)
    parser.add_argument("--opt", default="sgd", type=str, choices=("adam", "sgd"))
    parser.add_argument("--val", default="is", type=str, choices=("is", None))
    parser.add_argument("--train_prop", default=0.8, type=int)
    #parser.add_argument("--n_test", default=200, type=int)
    parser.add_argument("--n_epochs", default=2000, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--wd", default=5e-3, type=float)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--noise", default=0.5, type=float)
    parser.add_argument("--silent", default=False, action="store_true")
    parser.add_argument("--ratio_normalize", default=False, action="store_true")
    parser.add_argument("--dropout", default=0.0, type=float)

    # regularizations availables
    parser.add_argument("--no_erm", default=True, dest="erm", action="store_false")
    parser.add_argument("--var_reg", default=False, action="store_true")
    parser.add_argument("--ratio_reg", default=False, action="store_true")
    parser.add_argument("--combo_reg", default=False, action="store_true")
    parser.add_argument("--pos_reg", default=False, action="store_true")
    parser.add_argument("--pos_reg_tr", default=False, action="store_true")
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
