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
import logging

from tresnet import datamodules, shifts, glms, Tresnet

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    pl.seed_everything(123 * args.seed)

    # shift function, e.g., percent, subtract, cutoff
    shift = getattr(shifts, args.shift.capitalize())()
    shift_values = np.linspace(0.0, 0.5, num=args.num_shifts, dtype=np.float32).tolist()

    # glm family, e.g. Gaussian, Poisson, Bernoulli
    glm_family = getattr(glms, args.glm_family.capitalize())()

    # make dataset from available options
    datamodule_kwargs = dict(
        shift=shift,
        family=glm_family,
        shift_values=shift_values,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        noise_scale=args.noise_scale,
    )

    if args.dataset == "ihdp":
        datamodule = datamodules.IHDP(**datamodule_kwargs)
    elif args.dataset == "news":
        datamodule = datamodules.News(**datamodule_kwargs)
    elif args.dataset == "sim-B":
        datamodule = datamodules.SimB(**datamodule_kwargs)
    elif args.dataset == "sim-N":
        datamodule = datamodules.SimN(**datamodule_kwargs)
    elif args.dataset.startswith("tcga"):
        dosage_variant = int(args.dataset.split("-")[1])
        datamodule = datamodules.TCGA(
            **datamodule_kwargs, data_opts=dict(dosage_variant=dosage_variant)
        )
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    # set weight of targeted regularization
    n_train = len(datamodule.train_ix)
    tr_loss_weight = args.tr_loss_weight * n_train ** (-0.5)

    # make model
    model = Tresnet(
        in_dim=datamodule.covariates.shape[1],
        hidden_dim=args.hidden_dim,
        shift_values=shift_values,
        shift=shift,
        outcome_freeze=args.outcome_freeze,
        outcome_spline_degree=2,
        outcome_spline_knots=[0.33, 0.66],
        outcome_type=args.backbone,
        glm_family=glm_family,
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
        true_srf_train=datamodule.train_srf,
        true_srf_val=datamodule.val_srf,
        plot_every_n_epochs=args.plot_every_n_epochs,
        estimator=args.estimator,
    )

    # configure directory to save model results, delete contents if experiment
    logdir = f"logs/{args.logdir}/{args.dataset}/{args.glm_family}/{args.seed:06d}"
    if args.experiment is not None:
        logdir += f"/{args.experiment}"
        has_solution = os.path.exists(f"{logdir}/srf_estimates.csv")
        if args.clean and os.path.exists(logdir):
            if not has_solution or args.overwrite:
                shutil.rmtree(logdir)
            else:
                logging.warning(f"Skipping {logdir} because it has a solution")
                return  # do nothing

    # configure loggers
    tb_logger = TensorBoardLogger(
        save_dir=".",
        name=logdir,
        version="" if args.clean else None,
        default_hp_metric=False,
    )
    loggers = [tb_logger]
    if args.csv_logger:
        csv_logger = CSVLogger(
            save_dir=".",
            name=tb_logger.log_dir,
            flush_logs_every_n_steps=min(100, args.epochs),
            version="",
        )
        loggers.append(csv_logger)

    # configure best model checkpointing
    callbacks = []
    if args.best_metric is not None:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor=args.best_metric,
            mode="min",
            save_top_k=1,
            dirpath=tb_logger.log_dir,
            every_n_epochs=10,
            filename="best",
        )
        callbacks.append(checkpoint_callback)

    # train model
    trainer = pl.Trainer(
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        max_epochs=args.epochs,
        gradient_clip_val=1.0,
        callbacks=callbacks,
        log_every_n_steps=min(100, args.epochs),
        check_val_every_n_epoch=10,
        logger=loggers,
        enable_progress_bar=(not args.silent),
    )
    trainer.fit(model, datamodule)

    # load best model
    if args.best_metric is not None:
        ckpt_path = checkpoint_callback.best_model_path
        model = Tresnet.load_from_checkpoint(ckpt_path)

    # retrieve and safe last srf estimate
    # TODO return shift values
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
        true_train_srf=datamodule.train_srf,
        true_test_srf=datamodule.val_srf,
        fluctuation=model.fluct_param().detach().cpu().numpy(),
    )

    estimates = pd.DataFrame(estimates)
    estimates.to_csv(f"{tb_logger.log_dir}/srf_estimates.csv", index=False)

    # save args as yaml
    with open(f"{tb_logger.log_dir}/args.yaml", "w") as f:
        yaml.dump(vars(args), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--density_grid_size", default=10, type=int)
    dsets = ("ihdp", "news", "sim-B", "sim-N", "tcga-1", "tcga-2", "tcga-3")
    parser.add_argument("--dataset", default="ihdp", type=str, choices=dsets)
    parser.add_argument("--hidden_dim", default=50, type=int)
    parser.add_argument("--outcome_freeze", default=False, action="store_true")
    parser.add_argument("--ratio_freeze", default=False, action="store_true")
    shifts_ = ("percent",)
    parser.add_argument("--shift", default="percent", type=str, choices=shifts_)
    parser.add_argument("--num_shifts", default=10, type=int)
    backbones = ("vc", "causal_mlp", "dr")
    parser.add_argument("--backbone", default="vc", type=str, choices=backbones)
    glms_ = ("gaussian", "poisson", "bernoulli")
    parser.add_argument("--glm_family", default="gaussian", type=str, choices=glms_)
    rlosses = ("ps", "hybrid", "classifier", "multips")
    parser.add_argument("--ratio_loss", default="classifier", type=str, choices=rlosses)
    parser.add_argument("--label_smoothing", default=0.1, type=float)
    parser.add_argument("--tr", default=False, action="store_true")
    parser.add_argument("--epochs", default=2000, type=int)
    parser.add_argument("--clever", default=False, action="store_true")
    tr_types = ("discrete", "spline")
    parser.add_argument("--tr_type", default="discrete", type=str, choices=tr_types)
    parser.add_argument("--tr_weight_norm", default=False, action="store_true")
    parser.add_argument("--tr_loss_weight", default=20, type=float)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--weight_decay", default=5e-3, type=float)
    parser.add_argument("--dropout", default=0.0, type=float)
    optimizers = ("adam", "sgd")
    parser.add_argument("--optimizer", default="adam", type=str, choices=optimizers)
    parser.add_argument("--batch_size", default=10000, type=int)
    parser.add_argument("--noise_scale", type=float, default=0.1)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--plot_every_n_epochs", default=100, type=int)
    parser.add_argument("--silent", default=False, action="store_true")
    parser.add_argument("--logdir", default="runs", type=str)
    parser.add_argument(
        "--no_csv", dest="csv_logger", default=True, action="store_false"
    )
    parser.add_argument("--best_metric", default=None, type=str)
    estimators = ("ipw", "aipw", "outcome", "tr")
    parser.add_argument("--estimator", default=None, type=str, choices=estimators)
    parser.add_argument("--experiment", default=None, type=str)
    parser.add_argument("--clean", default=False, action="store_true")
    parser.add_argument("--overwrite", default=False, action="store_true")
    parser.add_argument("--unmonitor", default=False, action="store_true")

    args = parser.parse_args()

    # load config file if provided and set experiment name
    if args.experiment is not None:
        with open(f"configs/{args.experiment}.yaml", "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        for k, v in config.items():
            setattr(args, k, v)

    if args.unmonitor:  # overwrite config file
        args.best_metric = None

    main(args)
