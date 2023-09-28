import logging

import hydra
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from omegaconf import DictConfig

from tresnet import Tresnet, datamodules, glms, shifts

LOGGER = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(args: DictConfig) -> None:
    pl.seed_everything(123 * args.seed)

    # shift function, e.g., percent, subtract, cutoff
    shift = getattr(shifts, args.shift.type.capitalize())()
    shift_values = np.linspace(
        args.shift.min, args.shift.max, num=args.shift.num, dtype=np.float32
    ).tolist()

    # glm family, e.g. Gaussian, Poisson, Bernoulli
    glm_family = getattr(glms, args.data_generation.family.capitalize())()

    # make dataset from available options
    datamodule_kwargs = dict(
        shift=shift,
        family=glm_family,
        shift_values=shift_values,
        batch_size=args.training.batch_size,
        num_workers=args.training.num_workers,
        noise_scale=args.data_generation.noise_scale,
        outcome_scale=args.data_generation.outcome_scale,
    )

    if args.dataset == "ihdp":
        datamodule = datamodules.IHDP(**datamodule_kwargs)
    elif args.dataset == "ihdp-B":
        datamodule = datamodules.IHDPB(**datamodule_kwargs)
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
        raise ValueError(f"Unknown dataset '{args.dataset}'")

    # set weight of targeted regularization
    n_train = len(datamodule.train_ix)
    tr_loss_weight = args.tr.weight * n_train ** (-0.5)

    # make model
    model = Tresnet(
        in_dim=datamodule.covariates.shape[1],
        hidden_dim=args.body.hidden_dim,
        enc_hidden_layers=args.body.hidden_layers,
        shift_values=shift_values,
        shift=shift,
        outcome_freeze=args.outcome.freeze,
        outcome_spline_degree=2,
        outcome_spline_knots=[0.33, 0.66],
        outcome_type=args.outcome.backbone,
        glm_family=glm_family,
        ratio_freeze=args.treatment.freeze,
        ratio_spline_degree=2,
        ratio_spline_knots=[0.33, 0.66],
        ratio_label_smoothing=args.treatment.label_smoothing,
        ratio_grid_size=args.treatment.grid_size,
        ratio_loss=args.treatment.loss,
        ratio_loss_weight=args.treatment.weight,
        tr=(not args.tr.freeze),
        tr_loss_weight=tr_loss_weight,
        tr_clever=args.tr.clever,
        tr_param_type=args.tr.type,
        tr_spline_degree=2,
        tr_spline_knots=[0.33, 0.66],
        ratio_norm=args.tr.weight_norm,
        tr_tmle=args.tr.tmle,
        tr_opt_freq=args.training.tr_opt_freq,
        act=getattr(nn, args.activation),
        opt_lr=args.training.lr,
        opt_weight_decay=args.training.weight_decay,
        opt_optimizer=args.training.optimizer,
        dropout=args.training.dropout,
        true_srf_train=datamodule.train_srf,
        true_srf_val=datamodule.val_srf,
        plot_every_n_epochs=max(1, args.training.plot_every * args.training.epochs),
        estimator=args.estimator,
        finetune_after=int(args.training.finetune.after * args.training.epochs),
        finetune_mask_ratio=args.training.finetune.mask_ratio,
        finetune_freeze_nuisance=args.training.finetune.freeze_nuisance,
    )

    # configure directory to save model results, delete contents if experiment
    # get run dir from hydra
    logdir = HydraConfig.get().run.dir

    # configure loggers
    tb_logger = TensorBoardLogger(
        save_dir=".",
        name=logdir,
        version="",  # , if args.clean else None,
        default_hp_metric=False,
    )
    loggers = [tb_logger]
    if args.training.csv_logger:
        csv_logger = CSVLogger(
            save_dir=".",
            name=tb_logger.log_dir,
            flush_logs_every_n_steps=min(100, args.epochs),
            version="",
        )
        loggers.append(csv_logger)

    # configure best model checkpointing
    callbacks = []
    if args.training.monitor is not None:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor=args.training.monitor,
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
        max_epochs=args.training.epochs,
        # gradient_clip_val=1.0,
        callbacks=callbacks,
        log_every_n_steps=min(100, args.training.epochs),
        check_val_every_n_epoch=min(1, args.training.epochs // 100),
        logger=loggers,
        enable_progress_bar=args.training.progbar,
    )
    trainer.fit(model, datamodule)

    # load best model
    if args.training.monitor is not None:
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


if __name__ == "__main__":
    main()
