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

import matplotlib
matplotlib.use('agg') 

from tresnet.models import VCNet, RatioNet
#from dataset.dataset import get_iter, make_dataset, DATASETS, set_seed

from dataset.medicare import set_seed, get_iter, DataMedicare


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
        naaqs = np.array([30, 15, 12, 11, 10, 9, 8, 7, 6], dtype=np.float32)
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

 
    # make optimizer
    if args.opt == "adam":
        optimizer = torch.optim.Adam(optim_params, lr=3e-4)
    elif args.opt == "sgd":
        optimizer = torch.optim.SGD(
            optim_params, lr=args.lr, momentum=0.9, nesterov=True
        )

    best_loss, best_model, best_iter = 1e6, deepcopy(model), 0

    best_model.eval()

    # training loop
    train_loader = get_iter(train_data, batch_size=args.batch_size, shuffle=True)
    eps_lr = 0.25 / len(train_loader)

    for epoch in tqdm(range(args.n_epochs), disable=args.silent):
        # dict to store all the losses per batch
        losses = defaultdict(lambda: deque(maxlen=len(train_loader)))

        # iterate each batch
        # for _, item in enumerate(train_loader):
        for _, (t, x, y, offset) in tqdm(enumerate(train_loader), total=len(train_loader), disable=True):

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

            # 2. outcome loss
            lp = model_output["predicted_outcome"]
            y_hat = offset * torch.sigmoid(lp)
            outcome_loss = F.poisson_nll_loss(y_hat, y, log_input = False)
            losses["outcome_loss"].append(outcome_loss.item())
            total_loss = total_loss + outcome_loss

            losses["total_loss"].append(total_loss.item())

            total_loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 10.0)
            optimizer.step()

        # evaluation
        # if epoch == 0 or (epoch + 1) % args.eval_every == 0:
        if (epoch + 1) % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                # replace best model if improves
                if args.val in ("is", "test"):
                    M = test_data
                    t, x, y, offset = M["treatment"], M["covariates"], M["outcome"], M["offset"]
                    ix = torch.LongTensor(np.random.choice(len(t), size=50000)).to(dev)
                    t, x, y, offset = [u[ix] for u in (t, x, y, offset)]
                    output = model.forward(t, x)
                    z, lp = output["z"], output["predicted_outcome"]
                    y_hat = offset * torch.sigmoid(lp)
 
                    # poisson loglikelihood validation loss
                    val_loss = F.poisson_nll_loss(y_hat, y, log_input = False).item()
                    if val_loss < best_loss:
                        best_model = deepcopy(model)
                        best_model.eval()
                        best_loss = val_loss
                        best_iter = epoch
       
                elif args.val is None:
                    best_model = deepcopy(model)
                    best_model.eval()

                # obtain all evaluation metrics

                if not args.silent:
                    print("== Epoch: ", epoch, " ==")
                    print("Metrics:")
                    for k, vec in losses.items():
                        print(f"  {k}: {np.mean(vec):.4f}")

            model.train()\

    # load best model


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
    parser.add_argument("--val", default="test", type=str, choices=("is", "test", None))
    parser.add_argument("--n_train", default=500, type=int)
    parser.add_argument("--n_test", default=200, type=int)
    parser.add_argument("--n_epochs", default=20, type=int)
    parser.add_argument("--batch_size", default=4000, type=int)
    parser.add_argument("--wd", default=5e-3, type=float)
    parser.add_argument("--lr", default=1e-4, type=float) 
    parser.add_argument("--ls", default=0.1, type=float) 
    # parser.add_argument("--eps_lr", default=0.001, type=float) p
    parser.add_argument("--beta", default=0.1, type=float)
    # parser.add_argument("--noise", default=0.5, type=float)
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
