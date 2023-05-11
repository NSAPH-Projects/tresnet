import argparse

import numpy as np
import torch
import lightning.pytorch as pl

from dataset.datasets import make_dataset, DATASETS
from tresnet.tresnet import Tresnet


def main(args: argparse.Namespace) -> None:
    shift_values = np.linspace(0.0, 0.5, num=args.num_shifts, dtype=np.float32).tolist()

    # make dataset from available optionss
    D = make_dataset(
        args.dataset,
        shift_values,
        n_train=args.n_train,
        n_test=args.n_test,
        noise_scale=args.noise,
        count=args.count,
    )

    # get dataset size and confounder dimension
    n, covar_dim = D.train_matrix.shape[0], D.train_matrix.shape[1] - 2

    # set weight of targeted regularization
    args.tr_loss_weight = 5 * n ** (-0.5)

    # make model
    model = Tresnet(
        in_dim=covar_dim,
        hidden_dim=args.hidden_dim,
        shift_values=shift_values,
        shift_type=D.shift_type,
        outcome_head=(not args.no_outcome),
        outcome_spline_degree=2,
        outcome_spline_knots=[0.33, 0.66],
        outcome_type=args.outcome_type,
        outcome_family=args.loss_family,
        ratio_head=(not args.no_ratio),
        ratio_spline_degree=2,
        ratio_spline_knots=[0.33, 0.66],
        ratio_label_smoothing=args.label_smoothing,
        ratio_grid_size=args.density_grid_size,
    )

    trainer = pl.Trainer(
        device="cuda" if torch.cuda.is_available() else "cpu", max_epochs=args.n_epochs
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--density_grid_size", default=25, type=int)
    parser.add_argument("--dataset", default="sim-B", type=str, choices=DATASETS)
    parser.add_argument("--hidden_dim", default=50, type=int)
    parser.add_argument("--no_outcome", default=False, action="store_true")
    parser.add_argument("--no_ratio", default=False, action="store_true")
    parser.add_argument("--num_shifts", default=25, type=int)
    parser.add_argument(
        "--outcome_type", default="vc", type=str, choices=("vc", "mlp", "dr")
    )
    parser.add_argument(
        "--loss_family",
        default="gaussian",
        type=str,
        choices=("gaussian", "poisson", "bernoulli"),
    )

    # parser.add_argument(
    #     "--pert", default="simple", type=str, choices=("original", "simple")
    # )
    # parser.add_argument("--detach_ratio", default=False, action="store_true")
    # parser.add_argument("--rdir", default="results", type=str)
    # parser.add_argument("--edir", default=None, type=str)
    # parser.add_argument("--opt", default="sgd", type=str, choices=("adam", "sgd"))
    # parser.add_argument(
    #     "--val", default="none", type=str, choices=("is", "val", "none")
    # )
    # parser.add_argument("--n_train", default=500, type=int)
    # parser.add_argument("--n_test", default=200, type=int)
    parser.add_argument("--n_epochs", default=2000, type=int)
    # parser.add_argument("--batch_size", default=4000, type=int)
    # parser.add_argument("--eval_every", default=100, type=int)
    # parser.add_argument("--wd", default=1e-3, type=float)
    # parser.add_argument("--lr", default=1e-4, type=float)
    # parser.add_argument("--ls", default=0.1, type=float)

    # parser.add_argument("--beta", default=0.1, type=float)
    # parser.add_argument("--noise", default=0.1, type=float)  # it is 0.5 in vcnet paper
    # parser.add_argument("--silent", default=False, action="store_true")
    # parser.add_argument("--ratio_norm", default=True, action="store_true")
    # parser.add_argument("--dropout", default=0.2, type=float)

    # # regularizations availables
    # parser.add_argument(
    #     "--ratio", default="erm", type=str, choices=("erm", "gps_ratio", "c_ratio")
    # )
    # parser.add_argument("--var_reg", default=False, action="store_true")
    # parser.add_argument("--outcome_only", default=False, action="store_true")
    # parser.add_argument("--ratio_only", default=False, action="store_true")
    # parser.add_argument("--ratio_reg", default=False, action="store_true")
    # parser.add_argument("--combo_reg", default=False, action="store_true")
    # parser.add_argument("--pos_reg", default=False, action="store_true")
    # parser.add_argument("--pos_reg_tr", default=False, action="store_true")
    # parser.add_argument("--tr_reg", default=False, action="store_true")
    # parser.add_argument("--poisson", default=False, action="store_true")
    # parser.add_argument("--count", default=False, action="store_true")
    # parser.add_argument(
    #     "--backbone", default="vcnet", choices=("vcnet", "drnet", "mlp")
    # )
    # parser.add_argument("--target", type=str, default="si", choices=("si", "erf"))
    # parser.add_argument("--tr", default="discrete", choices=("discrete", "vc"))
    # parser.add_argument("--fit_ratio_scale", default=False, action="store_true")
    # parser.add_argument("--reg_multiscale", default=False, action="store_true")

    args = parser.parse_args()
    pl.seed_everything(123 * args.seed)

    main(args)
