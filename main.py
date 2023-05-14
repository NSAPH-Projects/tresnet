import argparse
import os
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
import yaml

from dataset.datasets import make_dataset, DATASETS
from tresnet.tresnet import Tresnet
from tresnet.datamodule import DataModule


def main(args: argparse.Namespace) -> None:
    pl.seed_everything(123 * args.seed % 1000000)

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
        outcome_freeze=args.outcome_freeze,
        outcome_spline_degree=2,
        outcome_spline_knots=[0.33, 0.66],
        outcome_type=args.backbone,
        outcome_family=args.family,
        ratio_freeze=args.ratio_freeze,
        ratio_spline_degree=2,
        ratio_spline_knots=[0.33, 0.66],
        ratio_label_smoothing=args.label_smoothing,
        ratio_grid_size=args.density_grid_size,
        ratio_loss=args.ratio_loss,
        ratio_loss_weight=1.0,
        tr=args.tr,
        tr_loss_weight=tr_loss_weight,
        tr_clever=args.clever,
        tr_param_type=args.tr_type,
        tr_spline_degree=2,
        tr_spline_knots=[0.33, 0.66],
        tr_weight_norm=args.tr_weight_norm,
        act=nn.SiLU,
        opt_lr=args.lr,
        opt_weight_decay=args.weight_decay,
        opt_optimizer=args.optimizer,
        dropout=args.dropout,
        true_srf_train=D["train_srf"],
        true_srf_val=D["test_srf"],
        plot_every_n_epochs=args.plot_every_n_epochs,
        estimator=args.estimator,
    )

    betches_per_epoch = datamodule.training_batches_per_epoch

    # configure directory to save model results, delete contents if experiment
    logdir = f"logs/{args.logdir}/{args.dataset}/{args.family}/{args.seed:06d}"
    if args.experiment is not None:
        logdir += f"/{args.experiment}"
        if args.clean and os.path.exists(logdir):
            shutil.rmtree(logdir)

    # configure loggers
    tb_logger = TensorBoardLogger(
        save_dir=".",
        name=logdir,
        version="" if args.clean else None,
        default_hp_metric=False,
    )
    csv_logger = CSVLogger(
        save_dir=".",
        name=tb_logger.log_dir,
        flush_logs_every_n_steps=betches_per_epoch,
        version="",
    )

    # configure best model checkpointing
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=args.best_metric,
        mode="min",
        save_top_k=1,
        dirpath=tb_logger.log_dir,
        every_n_epochs=10,
        filename="best",
    )

    # train model
    trainer = pl.Trainer(
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        max_epochs=args.epochs,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback],
        log_every_n_steps=betches_per_epoch,
        check_val_every_n_epoch=10,
        logger=[tb_logger, csv_logger],
        enable_progress_bar=(not args.silent),
    )
    trainer.fit(model, datamodule)

    # load best model
    if args.best_model:
        ckpt_path = checkpoint_callback.best_model_path
        model = Tresnet.load_from_checkpoint(ckpt_path)

    # retrieve and safe last srf estimate
    estimates = dict(
        train_srf=model.srf_estimator_train.detach().cpu().numpy(),
        test_srf=model.srf_estimator_val.detach().cpu().numpy(),
        train_srf_ipw=model.srf_ipw_train.detach().cpu().numpy(),
        test_srf_ipw=model.srf_ipw_val.detach().cpu().numpy(),
        train_srf_aipw=model.srf_aipw_train.detach().cpu().numpy(),
        test_srf_aipw=model.srf_aipw_val.detach().cpu().numpy(),
        train_srf_tr=model.srf_tr_train.detach().cpu().numpy(),
        test_srf_tr=model.srf_tr_val.detach().cpu().numpy(),
        train_srf_outcome=model.srf_outcome_train.detach().cpu().numpy(),
        test_srf_outcome=model.srf_outcome_val.detach().cpu().numpy(),
        true_srf_train=D["train_srf"],
        true_test_srf=D["test_srf"],
        fluctuation=model.fluct_param().detach().cpu().numpy(),
    )

    estimates = pd.DataFrame(estimates)
    estimates.to_csv(f"{tb_logger.log_dir}/srf_estimates.csv", index=False)

    # save model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--density_grid_size", default=10, type=int)
    parser.add_argument("--dataset", default="tcga-B", type=str, choices=DATASETS)
    parser.add_argument("--hidden_dim", default=50, type=int)
    parser.add_argument("--outcome_freeze", default=False, action="store_true")
    parser.add_argument("--ratio_freeze", default=False, action="store_true")
    parser.add_argument("--num_shifts", default=10, type=int)
    backbones = ("vc", "causal_mlp", "dr")
    parser.add_argument("--backbone", default="vc", type=str, choices=backbones)
    families = ("gaussian", "poisson", "bernoulli")
    parser.add_argument("--family", default="gaussian", type=str, choices=families)
    rlosses = ("ps", "hybrid", "classifier")
    parser.add_argument("--ratio_loss", default="classifier", type=str, choices=rlosses)
    parser.add_argument("--label_smoothing", default=0.1, type=float)
    parser.add_argument("--tr", default=False, action="store_true")
    parser.add_argument("--epochs", default=2000, type=int)
    parser.add_argument("--clever", default=False, action="store_true")
    tr_types = ("discrete", "spline")
    parser.add_argument("--tr_type", default="discrete", type=str, choices=tr_types)
    parser.add_argument("--tr_weight_norm", default=False, action="store_true")
    parser.add_argument("--tr_loss_weight", default=20, type=float)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--weight_decay", default=5e-3, type=float)
    parser.add_argument("--dropout", default=0.0, type=float)
    optimizers = ("adam", "sgd")
    parser.add_argument("--optimizer", default="adam", type=str, choices=optimizers)
    parser.add_argument("--batch_size", default=10000, type=int)
    parser.add_argument("--obs_noise", type=float, default=0.5)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--plot_every_n_epochs", default=100, type=int)
    parser.add_argument("--silent", default=False, action="store_true")
    parser.add_argument("--logdir", default="runs", type=str)
    parser.add_argument("--best_model", default=False, action="store_true")
    parser.add_argument("--best_metric", default="val/tr", type=str)
    estimators = ("ipw", "aipw", "outcome", "tr")
    parser.add_argument("--estimator", default=None, type=str, choices=estimators)
    parser.add_argument("--experiment", default=None, type=str)
    parser.add_argument("--clean", default=False, action="store_true")

    args = parser.parse_args()

    # load config file if provided and set experiment name
    if args.experiment is not None:
        with open(f"experiment_configs/{args.experiment}.yaml", "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        for k, v in config.items():
            setattr(args, k, v)

    main(args)
