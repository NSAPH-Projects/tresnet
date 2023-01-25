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
from tqdm import tqdm
import proplot as pplt

plt.ioff()
# from matplotlib import rc
# rc("pdf", fonttype=3)
# rc('font',**{'family':'serif'})
# rc('text', usetex=True)

from models.VCNet import VCNet, RatioNet
from utils import ratios
from models.modules import TargetedRegularizerCoeff
#from dataset.dataset import get_iter, make_dataset, DATASETS, set_seed

from dataset.dataset_medicare import set_seed, get_iter, DataMedicare


def main(args: argparse.Namespace) -> None:
    # seed and use gpu when available
    set_seed(1234 + 131 * args.seed)  # like torch manual seed at all levels
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # experiment_dir
    s1 = f"var_{int(args.var_reg)}"
    s2 = f"trreg_{int(args.tr_reg)}"
    s3 = f"pos_{int(args.pos_reg)}"
    s4 = f"dout_{args.dropout}"
    s5 = f"ratio_{args.ratio}"
    if args.edir is None:
        edir = f"{'-'.join([s1, s2, s3, s4, s5])}/{args.seed:02d}"
    else:
        edir = f"{args.edir}/{args.seed:02d}"
    os.makedirs(f"{args.rdir}/{args.dataset}/{edir}", exist_ok=True)

    # these are the shifting values use in the srf curve
    # steps = 10
    # delta_list = np.linspace(0.5 / steps, 0.5, num=steps, dtype=np.float32).tolist()
   

    # make dataset from available optionss
    path = "./data_medicare/zip_data.csv"
    treatment_col= "pm25"
    outcome_col= "dead"
    offset_col= "time_count"
    categorical_variables = ["regionNORTHEST", "regionSOUTH", "regionWEST"]
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

    shift_type = args.shift_type
    
    tmin, tmax = data.treatment_norm_min, data.treatment_norm_max
    if shift_type == "cutoff":
        naaqs = np.array([data.treatment_norm_max, 15, 12, 11, 10, 9, 8, 7, 6], dtype=np.float32)
        delta_list_unscaled = naaqs
        delta_list = (delta_list_unscaled - tmin) / (tmax - tmin)
    elif args.shift_type == "percent":
        naaqs = np.array([12, 11, 10, 9, 8], dtype=np.float32)
        quantiles = [0.99, 0.95, 0.9, 0.75, 0.5, 0.25]
        steps = 20
        t = torch.cat([train_data["treatment"], test_data["treatment"]])
        delta_list = np.linspace(0.0, 0.5, num=steps, dtype=np.float32).tolist()
        delta_list_unscaled = 100 *  np.array(delta_list)
        misaligned = np.zeros((len(delta_list), len(naaqs)))
        quantmat = np.zeros((len(delta_list), len(quantiles)))
        for i, d in enumerate(delta_list):
            shifted = ratios.shift(t, d, "percent")
            for j, th in enumerate(naaqs):
                misaligned[i, j] = (shifted > (th  - tmin) / (tmax - tmin)).float().mean().item()
            quantmat[i, :] = tmin + (tmax - tmin) * np.quantile(shifted.cpu().numpy(), quantiles)
        quantdf = pd.DataFrame(quantmat, columns=[f"q-{q}" for q in quantiles], index=delta_list_unscaled)
        quantdf.to_csv(f"{args.rdir}/{args.dataset}/{edir}/quants.csv")

        misaligneddf = pd.DataFrame(misaligned, columns=[f"naaqs-{h}" for h in naaqs], index=delta_list_unscaled)
        misaligneddf.to_csv(f"{args.rdir}/{args.dataset}/{edir}/misaligned.csv")

    # make neural network model
    density_estimator_config = [(input_dim, 50, 1), (50, 50, 1)]
    pred_head_config = [(50, 50, 1), (50, 1, 1)]

    if args.ratio != "c_ratio":
        model = VCNet(
            density_estimator_config,
            num_grids=args.n_grid,
            pred_head_config=pred_head_config,
            spline_degree=2,
            spline_knots=[0.33, 0.66],
            dropout=args.dropout,
        ).to(dev)
        density_head = model.density_estimator
    else:
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
            ls=args.ls
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
        if args.tr == "original":
            targeted_regularizer.requires_grad_(True)
            optim_params.append(
                {"params": targeted_regularizer, "momentum": 0.0, "weight_decay": 0.0}
            )
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
    train_loader = get_iter(train_data, batch_size=args.batch_size, shuffle=True)
    eps_lr = 1 / len(train_loader)

    for epoch in range(args.n_epochs):
        # dict to store all the losses per batch
        losses = defaultdict(lambda: deque(maxlen=len(train_loader)))

        # iterate each batch
        # for _, item in enumerate(train_loader):
        for _, (t, x, y, offset) in tqdm(enumerate(train_loader), total=len(train_loader)):

            # Now the offset is avilable as above, try it out 

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
            if args.ratio == "erm":
                probs = model_output["prob_score"]
                density_negll = -probs.log().mean()
                losses["density_negll"].append(density_negll.item())
                total_loss = total_loss + density_negll

            elif args.ratio == "gps_ratio":
                gps_ratio_loss = ratio_reg(t, density_head, z, shift_type)
                gps_ratio_loss = gps_ratio_loss + (1 / n) * ratio_reg.prior()
                total_loss = total_loss + gps_ratio_loss
                losses["gps_ratio_loss"].append(gps_ratio_loss.item())

            elif args.ratio == "c_ratio":
                c_ratio_losses = []
                for j, d in enumerate(delta_list):
                    t_d = ratios.shift(t, d, shift_type)
                    logits = torch.cat([model.log_ratio(t_d, z, j), model.log_ratio(t, z, j)])
                    tgts = torch.cat([torch.ones_like(t), torch.zeros_like(t)]).clamp(args.ls, 1 - args.ls)
                    L = F.binary_cross_entropy_with_logits(logits, tgts)
                    c_ratio_losses.append(L)
                c_ratio_loss = sum(c_ratio_losses) / len(delta_list)
                total_loss = total_loss + c_ratio_loss
                losses["c_ratio_loss"].append(c_ratio_loss.item())

            # 2. outcome loss
            lp = model_output["predicted_outcome"]
            y_hat = offset * F.softplus(lp)
            outcome_loss = F.poisson_nll_loss(y_hat, y, log_input = False)
            losses["outcome_loss"].append(outcome_loss.item())
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
                ratio = log_ratio.clamp(-5, 5).exp()
                ratio_ = ratio.detach() if args.detach_ratio else ratio
                if args.ratio_norm:
                    ratio_ = ratio_ / ratio_.mean()
                if args.pert == "simple":
                    y_pert = y_hat * eps.exp()
                    # L = (ratio_ * (y_pert - y).pow(2)).mean()
                    L = (ratio_ * F.poisson_nll_loss(y_pert, y, log_input=False, reduction='none')).mean()
                    with torch.no_grad():
                        bias = (ratio_ * y).mean().log() - (ratio_ * y_hat).mean().log()
                elif args.pert == "original":
                    y_pert = y_hat * torch.exp(ratio_ * eps)
                    # L = F.mse_loss(y_pert, y)
                    L = F.poisson_nll_loss(y_pert, y, log_input=False)
                    with torch.no_grad():
                        bias = 0.0  # (ratio_ * (y - y_hat)).mean() / ratio_.pow(2).mean()
                if args.tr == "discrete" and args.pert == "simple": # manual update of epsilon
                    biases[j] = bias
                    # eps.add_((bias - eps).item(), alpha=args.eps_lr)
                tr_losses.append(L)
            tr_losses = sum(tr_losses) / len(delta_list)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            if args.tr == 'discrete':
                targeted_regularizer += eps_lr * (biases - targeted_regularizer)
            # update epsilon (coordinate ascent)

        # evaluation
        if epoch == 0 or (epoch + 1) % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                # replace best model if improves
                if args.val == "is":
                    M = test_data
                    t, x, y, offset = M["treatment"], M["covariates"], M["outcome"], M["offset"]
                    ix = torch.LongTensor(np.random.choice(len(t), size=50000)).to(dev)
                    t, x, y, offset = [u[ix] for u in (t, x, y, offset)]
                    output = model.forward(t, x)
                    z, lp = output["z"], output["predicted_outcome"]
                    y_hat = offset * F.softplus(lp)
                    val_losses = []
                    for j, d in enumerate(delta_list):
                        if args.ratio != "c_ratio":
                            log_ratio = ratios.log_density_ratio_under_shift(
                                t=t,
                                delta=torch.full_like(t, d),
                                density_estimator=best_model.density_estimator,
                                z=z,
                                shift_type=shift_type,
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
                                y_pert = y_hat * torch.exp(eps * ratio)
                            elif args.pert == "simple":
                                y_pert= y_hat * eps.exp()
                        else:
                            y_pert = y_hat
                        val_losses.append((ratio * (y_pert - y).pow(2)).mean().item())
                    val_loss = float(np.mean(val_losses))
                    if val_loss < best_loss:
                        best_model = deepcopy(model)
                        best_model.eval()
                        best_loss = val_loss
                        best_iter = epoch
                        best_tr = deepcopy(targeted_regularizer) if args.tr == "vc" else targeted_regularizer.clone()

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
                    t, x, y, offset = M["treatment"], M["covariates"], M["outcome"], M["offset"]
                    ix = torch.LongTensor(np.random.choice(len(t), size=50000)).to(dev)
                    t, x, y, offset = [u[ix] for u in (t, x, y, offset)]
                    best_model_output = best_model.forward(t, x)
                    z = best_model_output["z"]
                    lp = best_model_output['predicted_outcome']
                    y_hat = offset * F.softplus(lp)

                    # dictionaries for all kind of estimates
                    # ipw_estims = []
                    # ipw_errors = []
                    aipw_estims = []
                    # tmle_estims = []
                    # tmle_errors = []
                    tr_estims = []
                    plugin_estims = []

                    for j, d in enumerate(delta_list):
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
                        # estim = (ratio * y).mean().item()
                        # error = (estim - truth).item()
                        # ipw_estims.append(estim)
                        # ipw_errors.append(error)

                        # A-IPTW estimates
                        t_delta = ratios.shift(t, d, shift_type)
                        y_delta = offset * model.prediction_head(t_delta, z).exp()
                        if args.tr == "discrete":
                            eps = best_tr[j]
                        elif args.tr == "vc":
                            eps = best_tr(torch.full_like(t, d))
                        if args.pert == "original":
                            y_pert = y_hat * torch.exp(eps * ratio)
                            y_pert_delta = y_delta * torch.exp(eps * ratio)
                        elif args.pert == "simple":
                            y_pert = y_hat * eps.exp()
                            y_pert_delta = y_delta * eps.exp()
                        estim = (ratio * (y - y_pert)).mean() + y_pert_delta.mean()
                        aipw_estims.append(estim.item())

                        # Targeted Regularization
                        estim = y_pert_delta.mean()
                        tr_estims.append(estim.item())

                        # Plugin
                        estim = y_delta.mean()
                        plugin_estims.append(estim.item())

                    # add estimation error as columns of result dataframe
                    # df[part + "_ipw_estim"] = ipw_estims
                    # df[part + "_ipw_error"] = ipw_errors
                    df[part + "_aipw_estim"] = aipw_estims
                    df[part + "_tr_estim"] = tr_estims
                    df[part + "_plugin_estim"] = plugin_estims

                    # save metrics #TODO: this is only doing test, must upate to use df
                    # for computation
                    # TODO, separate test and train metrics
                    metrics = {k: float(np.mean(v)) for k, v in losses.items()}
                    # metrics["ipw_curve_error"] = float(
                    #     np.square(ipw_errors).mean() ** 0.5
                    # )
                    metrics_path = (
                        f"{args.rdir}/{args.dataset}/{edir}/metrics_{part}.yaml"
                    )
                    metrics["last_saved_epoch"] = epoch
                    metrics["best_iter"] = best_iter
                    metrics["best_val"] = best_loss
                    with open(metrics_path, "w") as io:
                        yaml.safe_dump(metrics, io)

                # save estimated curve dataset
                results_path = f"{args.rdir}/{args.dataset}/{edir}/curve.csv"
                df.round(4).to_csv(results_path, index=False)

                # save experiment config
                config_path = f"{args.rdir}/{args.dataset}/{edir}/config.yaml"
                with open(config_path, "w") as io:
                    yaml.safe_dump(vars(args), io)

                # plot curves
                if shift_type == "cutoff":
                    # _, ax = pplt.subplots([1, 2], figsize=(7, 3), wspace=5)
                    _, ax = pplt.subplots([1], figsize=(3.5, 2.5))
                elif shift_type == "percent":
                    # _, ax = pplt.subplots([[1, 2], [3, 4]], figsize=(7, 5), hspace=1, wspace=5, sharey=False)
                    _, ax = pplt.subplots([[1, 2, 3]], figsize=(9, 3), hspace=1, wspace=5, share=False)
                # plt.subplots_adjust(wspace=0.5, hspace=0.5)
                # for now make a weighted average, whiles we figure out evaluating the entire data
                pct_tr = df.train_tr_estim / df.train_tr_estim.iloc[0] - 1
                pct_val = df.test_tr_estim / df.test_tr_estim.iloc[0] - 1
                N_tr, N_val = train_data["treatment"].shape[0], test_data["treatment"].shape[0]
                pct = (N_tr * pct_tr + N_val * pct_val) / (N_tr + N_val)
                ax[0].plot(
                    delta_list_unscaled, 100 * pct, c="blue", ls="--"
                )
               
                # ax[1].plot(
                #     delta_list_unscaled, 100 * pct_val, label="Test", c="red", ls="--"
                # )
                fig_path = f"{args.rdir}/{args.dataset}/{edir}/fig.png"
                ax[0].legend(framealpha=0.5); ax[0].set_ylabel("Reduction in deaths (%)");
                # ax[1].legend(); ax[1].set_ylabel("Reduction in deaths (%)"); 
                
                if shift_type == "cutoff":
                    ax[0].set_xlim(6, 15)
                    ax[0].set_xlabel("NAAQS cutoff")
                    # ax[1].set_xlim(8, 14)
                    # for k in range(2):
                    #     ax[k].set_xlabel("NAAQS")

                if shift_type == "percent":
                    # plut naaq compliance
                    ax[1].plot(delta_list_unscaled, misaligned, label=naaqs.astype(int))
                    ax[1].set_ylabel("Zipcode-years above threshold (%)"); 
                    ax[1].legend(title="NAAQS threshold", framealpha=0.5)
                    ax[2].plot(delta_list_unscaled, quantmat, label=[f"{int(100*q)}%" for q in quantiles])
                    ax[2].set_ylabel("$PM_{2.5}$ ($\mu g/m^3$)"); 
                    ax[2].legend(title="Quantiles", framealpha=0.5)
                    for k in range(3):
                        ax[k].set_xlabel("$PM_{2.5}$ reduction (%)")
                
                plt.savefig(fig_path, bbox_inches='tight')
                plt.close()

            model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--n_grid", default=30, type=int)
    parser.add_argument(
        "--pert", default="simple", type=str, choices=("original", "simple")
    )
    parser.add_argument("--detach_ratio", default=False, action="store_true")
    parser.add_argument("--eval_every", default=10, type=int)
    parser.add_argument("--rdir", default="results", type=str)
    parser.add_argument("--edir", default=None, type=str)
    parser.add_argument("--opt", default="sgd", type=str, choices=("adam", "sgd"))
    parser.add_argument("--val", default="is", type=str, choices=("is", None))
    parser.add_argument("--n_train", default=500, type=int)
    parser.add_argument("--n_test", default=200, type=int)
    parser.add_argument("--n_epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=500, type=int)
    parser.add_argument("--wd", default=3e-5, type=float)
    parser.add_argument("--lr", default=1e-4, type=float) 
    parser.add_argument("--ls", default=0.0, type=float) 
    # parser.add_argument("--eps_lr", default=0.001, type=float) p
    parser.add_argument("--beta", default=0.1, type=float)
    parser.add_argument("--noise", default=0.5, type=float)
    parser.add_argument("--train_prop", default=0.8, type=float)
    parser.add_argument("--silent", default=False, action="store_true")
    parser.add_argument("--ratio_norm", default=True, action="store_true")
    parser.add_argument("--dropout", default=0.05, type=float)
    parser.add_argument("--mc_dropout", default=False, action="store_true")

    # regularizations availables
    parser.add_argument("--ratio", default="c_ratio", type=str, choices=("erm", "gps_ratio", "c_ratio"))
    parser.add_argument("--shift_type", default="cutoff", type=str, choices=("percent", "cutoff"))
    parser.add_argument("--var_reg", default=False, action="store_true")
    parser.add_argument("--ratio_reg", default=False, action="store_true")
    parser.add_argument("--combo_reg", default=False, action="store_true")
    parser.add_argument("--pos_reg", default=False, action="store_true")
    parser.add_argument("--pos_reg_tr", default=False, action="store_true")
    parser.add_argument("--tr_reg", default=True, action="store_true")
    parser.add_argument("--tr", default="discrete", choices=("discrete", "vc"))
    parser.add_argument("--fit_ratio_scale", default=False, action="store_true")
    parser.add_argument("--reg_multiscale", default=False, action="store_true")

    args = parser.parse_args()
    args.dataset = "medicare"

    # with torch.autograd.set_detect_anomaly(True):
    main(args)
