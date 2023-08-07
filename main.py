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

from tresnet import Tresnet, glms, shifts

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
    glm_family = getattr(glms, args.family.capitalize())()

    # make dataset from available options
    datamodule_kwargs = dict(
        shift=shift,
        family=glm_family,
        shift_values=shift_values,
        batch_size=args.training.batch_size,
        num_workers=args.training.num_workers,
        # noise_scale=args.data_generation.noise_scale,
        # outcome_scale=args.data_generation.outcome_scale,
        shuffle_batches=args.training.shuffle_batches,
    )

    datamodule = hydra.utils.instantiate(args.dataset, **datamodule_kwargs)

    # set weight of targeted regularization
    n_train = len(datamodule.train_ix)
    tr_loss_weight = args.tr.base_weight * n_train ** (-0.5)

    # make model
    model = Tresnet(
        in_dim=datamodule.covariates.shape[1],
        hidden_dim=args.body.hidden_dim,
        enc_hidden_layers=args.body.hidden_layers,
        independent_encoders=args.body.independent_encoders,
        shift_values=shift_values,
        shift=shift,
        outcome_freeze=args.outcome.freeze,
        outcome_spline_degree=2,
        outcome_spline_knots=[0.33, 0.66],
        outcome_type=args.outcome.backbone,
        outcome_loss_weight=args.outcome.weight,
        outcome_training_noise=args.outcome.training_noise,
        glm_family=glm_family,
        ratio_freeze=args.treatment.freeze,
        ratio_spline_degree=2,
        ratio_spline_knots=[0.33, 0.66],
        ratio_label_smoothing=args.treatment.label_smoothing,
        ratio_grid_size=args.treatment.grid_size,
        ratio_loss=args.treatment.loss,
        ratio_loss_weight=args.treatment.weight,
        ratio_norm_weight=args.treatment.norm_weight,
        ratio_norm=args.treatment.norm,
        tr=(not args.tr.freeze),
        tr_loss_weight=tr_loss_weight,
        tr_clever=args.tr.clever,
        tr_param_type=args.tr.type,
        tr_spline_degree=args.tr.spline_degree,
        tr_spline_knots=list(args.tr.spline_knots),
        tr_tmle=args.tr.tmle,
        tr_opt_freq=args.training.tr_opt_freq,
        tr_consistency_weight=args.tr.consistency * tr_loss_weight,
        act=getattr(nn, args.activation),
        optimizer=args.optimizer.name,
        optimizer_opts=args.optimizer.args,
        dropout=args.training.dropout,
        grad_clip=args.training.grad_clip,
        true_srf_train=datamodule.train_srf,
        true_srf_val=datamodule.val_srf,
        plot_every_n_epochs=max(1, args.training.plot_every * args.training.epochs),
        estimator=args.estimator,
        estimator_ma_weight=args.estimator_ma_weight,
        finetune_after=int(args.training.finetune.after * args.training.epochs),
        finetune_mask_ratio=args.training.finetune.mask_ratio,
        finetune_freeze_nuisance=args.training.finetune.freeze_nuisance,
        finetune_decrease_lr_after=int(
            args.training.finetune.decrease_lr_after * args.training.epochs
        ),
    )
    if args.compile:
        model = torch.compile(model)

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
    loggers = [tb_logger] if args.loggers.tb else []
    if args.loggers.csv:
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
        callbacks=callbacks,
        log_every_n_steps=min(100, args.training.epochs),
        check_val_every_n_epoch=min(1, args.training.epochs // 100),
        logger=loggers,
        enable_progress_bar=args.training.progbar,
        enable_checkpointing=False,
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
    )
    if args.tr and not args.tr.type == "erf":
        estimates["fluctuation"] = model.fluct_param().detach().cpu().numpy()

    estimates = pd.DataFrame(estimates)
    estimates.to_csv(f"{tb_logger.log_dir}/srf_estimates.csv", index=False)


if __name__ == "__main__":
    main()
