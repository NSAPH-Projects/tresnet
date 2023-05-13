import argparse

import numpy as np
import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

from dataset.datasets import make_dataset, DATASETS
from tresnet.tresnet import Tresnet
from tresnet.datamodule import DataModule


def main(args: argparse.Namespace) -> None:
    pl.seed_everything(123 * args.seed)

    shift_values = np.linspace(0.0, 0.5, num=args.num_shifts, dtype=np.float32).tolist()

    # make dataset from available optionss
    D = make_dataset(
        args.dataset,
        shift_values,
        noise_scale=args.obs_noise,
    )
    datamodule = DataModule(
        train_matrix=D["train_matrix"],
        test_matrix=D["test_matrix"],
        train_batch_size=args.batch_size,
        test_batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # set weight of targeted regularization
    tr_loss_weight = args.tr_loss_weight * datamodule.n_train ** (-0.5)

    # make model
    model = Tresnet(
        in_dim=datamodule.n_covariates,
        hidden_dim=args.hidden_dim,
        shift_values=shift_values,
        shift_type=D["shift_type"],
        outcome_head=(not args.no_outcome),
        outcome_spline_degree=2,
        outcome_spline_knots=[0.33, 0.66],
        outcome_type=args.backbone,
        outcome_family=args.loss_family,
        ratio_head=(not args.no_ratio),
        ratio_spline_degree=2,
        ratio_spline_knots=[0.33, 0.66],
        ratio_label_smoothing=args.label_smoothing,
        ratio_grid_size=args.density_grid_size,
        ratio_type=args.ratio_type,
        ratio_loss_weight=1.0,
        tr=args.tr,
        tr_loss_weight=tr_loss_weight,
        tr_use_clever=args.use_clever,
        tr_param_type=args.tr_type,
        tr_spline_degree=2,
        tr_spline_knots=[0.33, 0.66],
        tr_weight_norm=args.tr_weight_norm,
        act=nn.SiLU,
        opt_lr=args.lr,
        opt_weight_decay=args.weight_decay,
        opt_optimizer=args.optimizer,
        dropout=args.dropout,
        true_train_srf=D["srf_train"],
        true_val_srf=D["srf_test"],
        plot_every_n_epochs=args.plot_every_n_epochs,
    )

    betches_per_epoch = datamodule.training_batches_per_epoch
    loggers = [
        TensorBoardLogger(save_dir=".", name="lightning_logs/tb"),
        CSVLogger(
            save_dir=".",
            name="lightning_logs/csv",
            flush_logs_every_n_steps=betches_per_epoch,
        ),
    ]
    trainer = pl.Trainer(
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        max_epochs=args.n_epochs,
        gradient_clip_val=1.0,
        enable_checkpointing=False,
        log_every_n_steps=betches_per_epoch,
        check_val_every_n_epoch=10,
        logger=loggers,
    )
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--density_grid_size", default=10, type=int)
    parser.add_argument("--dataset", default="tcga-B", type=str, choices=DATASETS)
    parser.add_argument("--hidden_dim", default=50, type=int)
    parser.add_argument("--no_outcome", default=False, action="store_true")
    parser.add_argument("--no_ratio", default=False, action="store_true")
    parser.add_argument("--num_shifts", default=10, type=int)
    backbones = ("vc", "causal_mlp", "dr")
    parser.add_argument("--backbone", default="vc", type=str, choices=backbones)
    loss_families = ("gaussian", "poisson", "bernoulli")
    parser.add_argument(
        "--loss_family", default="gaussian", type=str, choices=loss_families
    )
    ratio_types = ("ps", "hybrid", "classifier")
    parser.add_argument("--ratio_type", default="ps", type=str, choices=ratio_types)
    parser.add_argument("--label_smoothing", default=0.1, type=float)
    parser.add_argument("--tr", default=False, action="store_true")
    parser.add_argument("--n_epochs", default=2000, type=int)
    parser.add_argument("--use_clever", default=False, action="store_true")
    tr_types = ("discrete", "spline")
    parser.add_argument("--tr_type", default="discrete", type=str, choices=tr_types)
    parser.add_argument("--tr_weight_norm", default=False, action="store_true")
    parser.add_argument("--tr_loss_weight", default=20, type=float)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--weight_decay", default=5e-3, type=float)
    parser.add_argument("--dropout", default=0.0, type=float)
    optimizers = ("adam", "sgd")
    parser.add_argument("--optimizer", default="adam", type=str, choices=optimizers)
    parser.add_argument("--batch_size", default=5000, type=int)
    parser.add_argument("--obs_noise", type=float, default=0.5)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--plot_every_n_epochs", default=100, type=int)

    args = parser.parse_args()

    main(args)
