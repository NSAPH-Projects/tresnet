import logging
from collections import defaultdict
from typing import Literal

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tresnet import glms, layers, shifts

# import matplotlib

LOGGER = logging.getLogger(__name__)

# matplotlib.use("Agg")


class OutcomeHead(nn.Module):
    """Outcome head model"""

    def __init__(
        self,
        outcome_type: Literal["vc", "causalmlp", "piecewise"],
        config: layers.ModuleConfig,
        vc_spline_degree: int = 2,
        vc_spline_knots: list[float] = [0.33, 0.66],
        glm_family: glms.GLMFamily = glms.Gaussian(),
    ) -> None:
        super().__init__()
        self.outcome_type = outcome_type
        self.glm_family = glm_family
        if outcome_type == "vc":
            kwargs = dict(spline_degree=vc_spline_degree, spline_knots=vc_spline_knots)
        elif outcome_type == "causalmlp":
            kwargs = dict()  # add dimension
        elif outcome_type == "piecewise":
            kwargs = dict()
        else:
            raise NotImplementedError
        self.model = config.make_module(outcome_type, **kwargs)
        self.intercept = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(
        self, treatment: Tensor, features: Tensor, detach_bias: bool = False
    ) -> Tensor:
        inputs = torch.cat([treatment[:, None], features], 1)
        bias = self.intercept if not detach_bias else self.intercept.detach()
        return self.model(inputs) + bias

    def loss(
        self,
        treatment: Tensor,
        features: Tensor,
        targets: Tensor,
        bias: Tensor | float = 0.0,
        weights: Tensor | None = None,
        detach_intercept: bool = False,
        return_errors: bool = False,
        noise: float = 0.0,
    ) -> Tensor:
        # obtain predictor
        if len(treatment.shape) == 1:
            lp = self(treatment, features, detach_bias=detach_intercept) + bias
        else:
            # reshape teratment long format
            assert len(treatment.shape) == len(bias.shape) == 2
            d = treatment.shape[1]
            treat_ = treatment.view(-1)
            feat_ = torch.cat([features] * d, 0)
            lp = self(treat_, feat_, detach_bias=detach_intercept).view(-1, d) + bias
        self.lp = lp

        # because pred is either has either one column for the outcom eloss
        # or n=len(shift_values) columns for the targeted regularization
        targets = targets[:, None].repeat(1, lp.shape[1])

        # eval loss per item
        lp_noisy = lp + noise * torch.randn_like(lp)
        loss_ = self.glm_family.loss(lp_noisy, targets, reduction="none")
        # aggregate
        if weights is not None:
            loss_ = loss_ * weights
        loss_ = loss_.mean()

        if not return_errors:
            return loss_
        else:
            link = self.glm_family.link
            errors = targets - link(lp)
            return loss_, errors


class RatioHead(nn.Module):
    """Ratio head model"""

    def __init__(
        self,
        shift_values: list[float],
        ratio_loss: Literal["ps", "hybrid", "multips", "classifier", "telescope"],
        shift: shifts.Shift,
        in_dim: int,
        ratio_grid_size: int,
        ratio_spline_degree: int = 2,
        ratio_spline_knots: list[float] = [0.33, 0.66],
        label_smoothing: float = 0.01,
        treatment_noise: float = 0.1,
    ) -> None:
        super().__init__()
        self.ratio_loss = ratio_loss
        self.register_buffer("shift_values", torch.FloatTensor(shift_values))
        self.label_smoothing = label_smoothing
        self.shift = shift
        self.treatment_noise = treatment_noise

        # validate shift type and ratio type
        if not ratio_loss in ("ps", "hybrid"):
            if not self.shift.has_inverse():
                raise ValueError("shift function must have inverse and logdet")

        # ratio model
        if ratio_loss in ("ps", "hybrid"):
            self.ps = layers.DiscreteDensityEstimator(in_dim, ratio_grid_size)
        elif ratio_loss == "multips":
            self.ps = layers.DiscreteDensityEstimator(in_dim, ratio_grid_size)
            self.multips = nn.ModuleList()
            for i in range(len(self.shift_values)):
                self.multips.append(
                    layers.DiscreteDensityEstimator(in_dim, ratio_grid_size)
                )
        elif ratio_loss in ("classifier", "telescope"):
            # classifier with num_shifts heads
            args = [in_dim, 1, ratio_spline_degree, ratio_spline_knots]
            self.class_logits = nn.ModuleList(
                [layers.VCLinear(*args) for _ in range(len(self.shift_values))]
            )
        else:
            raise NotImplementedError(f"ratio loss {ratio_loss} not implemented")

    def forward(self, treatment: Tensor, features: Tensor) -> Tensor:
        # there's two cases two handle, when treatment is a vector
        # and where treatment is a column, each column has been shifted
        # we want to be smart about broadcasting along shifts
        shift_values = self.shift_values[None, :]
        if len(treatment.shape) == 1:
            treatment = treatment[:, None].repeat(1, len(self.shift_values))

        if self.ratio_loss in ("ps", "hybrid"):
            ps_inv = []
            ps_obs = []
            inv, logdet = self.shift.inverse(treatment, shift_values)
            for i in range(len(self.shift_values)):
                inputs = torch.cat([inv[:, i, None], features], 1)
                ps_inv.append(self.ps(inputs))
                inputs = torch.cat([treatment[:, i, None], features], 1)
                ps_obs.append(self.ps(inputs))
            ps_inv = torch.stack(ps_inv, 1)
            ps_obs = torch.stack(ps_obs, 1)
            numerator = torch.log(ps_inv + 1e-2) + logdet
            denominator = torch.log(ps_obs + 1e-2)
            log_ratio = numerator - denominator

        elif self.ratio_loss == "multips":
            ps_shift = []
            ps_obs = []
            shifted = self.shift(treatment, shift_values)
            for i in range(len(self.shift_values)):
                inputs = torch.cat([shifted[:, i, None], features], 1)
                ps_shift.append(self.multips[i](inputs))
                inputs = torch.cat([treatment[:, i, None], features], 1)
                ps_obs.append(self.ps(inputs))
            ps_shift = torch.stack(ps_shift, 1)
            ps_obs = torch.stack(ps_obs, 1)
            numerator = torch.log(ps_shift + 1e-6)
            denominator = torch.log(ps_obs + 1e-6)
            log_ratio = numerator - denominator
        elif self.ratio_loss in ("classifier", "telescope"):
            log_ratio = []
            for i in range(len(self.shift_values)):
                inputs = torch.cat([treatment[:, i, None], features], 1)
                log_ratio.append(self.class_logits[i](inputs))
            log_ratio = 0.5 * torch.cat(log_ratio, 1)  # 0.1 for inductive bias
            log_ratio = 3.0 * torch.tanh((log_ratio).clamp(-3, 3))
            if self.ratio_loss == "telescope":
                log_ratio = torch.cumsum(log_ratio, dim=1)

        return log_ratio

    def loss(self, treatment: Tensor, features: Tensor) -> Tensor:
        if self.treatment_noise > 0:
            noise = self.treatment_noise * torch.randn_like(treatment)
            new_logits = torch.logit(treatment.clamp(0.01, 0.99)) + noise
            treatment = torch.sigmoid(new_logits)
        inputs = torch.cat([treatment[:, None], features], 1)
        if self.ratio_loss == "ps":
            # likelihood/erm loss
            ps_obs = self.ps(inputs)
            loss_ = -torch.log(ps_obs + 1e-6).mean()

        elif self.ratio_loss == "multips":
            ps_obs = self.ps(inputs)
            loss_ = -torch.log(ps_obs + 1e-6).mean()
            shifted = self.shift(treatment[:, None], self.shift_values[None, :])
            ps_shifted = [
                self.multips[i](torch.cat([shifted[:, i, None], features], 1))
                for i in range(len(self.shift_values))
            ]
            ps_shifted = torch.stack(ps_shifted, 1)
            loss_ = loss_ - torch.log(ps_shifted + 1e-6).mean()

        elif self.ratio_loss in ("hybrid", "classifier"):
            # classifier loss, but compute ratio from ps
            shifted = self.shift(treatment[:, None], self.shift_values[None, :])
            ratio1 = self(shifted, features)
            ratio2 = self(treatment, features)
            logits = torch.cat([ratio2, ratio1])
            tgts = torch.cat([torch.zeros_like(ratio2), torch.ones_like(ratio1)])
            tgts = tgts.clamp(self.label_smoothing / 2, 1 - self.label_smoothing / 2)
            loss_ = F.binary_cross_entropy_with_logits(logits, tgts)
        elif self.ratio_loss == "telescope":
            # telescope loss
            shifted = self.shift(treatment[:, None], self.shift_values[None, :])
            # eval for each shift
            logits_zeros = []
            logits_ones = []
            for i in range(len(self.shift_values)):
                if i == 0:
                    inputs_zeros = torch.cat([treatment[:, None], features], 1)
                else:
                    inputs_zeros = torch.cat([shifted[:, i - 1, None], features], 1)
                inputs_ones = torch.cat([shifted[:, i, None], features], 1)
                logits_zeros.append(self.class_logits[i](inputs_zeros))
                logits_ones.append(self.class_logits[i](inputs_ones))

            # make sure the tanh operation here is compatible with forward method!
            logits_zeros = 0.5 * torch.cat(logits_zeros, 1)
            logits_ones = 0.5 * torch.cat(logits_ones, 1)
            logits_zeros = 3.0 * torch.tanh(logits_zeros.clamp(-3, 3))
            logits_ones = 3.0 * torch.tanh(logits_ones.clamp(-3, 3))

            tgts = torch.cat(
                [torch.zeros_like(logits_zeros), torch.ones_like(logits_ones)]
            )
            tgts = tgts.clamp(self.label_smoothing / 2, 1 - self.label_smoothing / 2)
            logits = torch.cat([logits_zeros, logits_ones])
            loss_ = F.binary_cross_entropy_with_logits(logits, tgts)

        return loss_


class Tresnet(pl.LightningModule):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        enc_hidden_layers: int,
        shift_values: list[float],
        shift: shifts.Shift,
        independent_encoders: bool = False,
        outcome_freeze: bool = False,
        outcome_type: str = Literal["vc", "mlp", "piecewise"],
        outcome_spline_degree: int = 2,
        outcome_spline_knots: list[float] = [0.33, 0.66],
        outcome_loss_weight: float = 1.0,
        outcome_training_noise: float = 0.0,
        glm_family: glms.GLMFamily = glms.Gaussian(),
        ratio_freeze: bool = False,
        ratio_loss: Literal["ps", "hybrid", "classifier", "multips"] = "ps",
        ratio_spline_degree: int = 2,
        ratio_spline_knots: list[float] = [0.33, 0.66],
        ratio_grid_size: int = 10,
        ratio_label_smoothing: float = 0.01,
        ratio_loss_weight: float = 1.0,
        ratio_norm: bool = False,
        ratio_norm_weight: float = 0.0,
        tr: bool = True,
        tr_spline_degree: int = 2,
        tr_spline_knots: list[float] = list(np.linspace(0, 1, num=10)[1:-1]),
        tr_param_type: Literal["discrete", "splines"] = "discrete",
        tr_opt_freq: int = 100,
        tr_clever: bool = True,
        tr_loss_weight: float = 0.1,
        tr_consistency_weight: float = 0.0,
        tr_tmle: bool = False,
        act: nn.Module = nn.ReLU,
        optimizer: Literal["adam", "sgd"] = "adam",
        optimizer_opts: dict = {},
        dropout: float = 0.0,
        grad_clip: float = 1.0,
        true_srf_train: Tensor | None = None,
        true_srf_val: Tensor | None = None,
        plot_every_n_epochs: int = 100,
        estimator: None | Literal["ipw", "aipw", "outcome", "tr", "tr_aipw"] = None,
        estimator_ma_weight: float = 0.1,
        finetune_after: float | None = None,
        finetune_mask_ratio: float = 0.0,
        finetune_freeze_nuisance: bool = False,
        finetune_decrease_lr_after: int | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.register_buffer("shift_values", torch.FloatTensor(shift_values))
        self.outcome_freeze = outcome_freeze
        self.outcome_loss_weight = outcome_loss_weight
        self.outcome_training_noise = outcome_training_noise
        self.glm_family = glm_family
        self.ratio_freeze = ratio_freeze
        self.ratio_loss_weight = ratio_loss_weight
        self.ratio_norm_loss_weight = ratio_norm_weight
        self.tr = tr
        self.tr_param_type = tr_param_type
        self.tr_clever = tr_clever
        self.ratio_norm = ratio_norm
        self.tr_loss_weight = tr_loss_weight
        self.tr_freq = tr_opt_freq
        self.tr_tmle = tr_tmle
        self.tr_consistency_weight = tr_consistency_weight
        self.tr_erf = tr_param_type == "erf"
        self.optimizer = optimizer
        self.optimizer_opts = optimizer_opts
        self.register_buffer("true_srf_train", true_srf_train)
        self.register_buffer("true_srf_val", true_srf_val)
        self.shift = shift
        self.plot_every_n_epochs = plot_every_n_epochs
        self.estimator = estimator
        self.estimator_ma_weight = estimator_ma_weight
        self.finetune_after = finetune_after
        self.finetune_mask_ratio = finetune_mask_ratio
        self.finetune_freeze_nuisance = finetune_freeze_nuisance
        self.finetune_decrease_lr_after = finetune_decrease_lr_after
        self.grad_clip = grad_clip
        self.independent_encoders = independent_encoders

        # layer kwargs
        lkwargs = dict(
            act=act,
            dropout=dropout,
        )

        # make feature encoder
        hidden = []
        for _ in range(enc_hidden_layers):
            hidden.append(layers.LayerConfig(hidden_dim, hidden_dim, True, **lkwargs))
        encoder_config = layers.ModuleConfig(
            layers.LayerConfig(in_dim, hidden_dim, True, **lkwargs),
            *hidden,
        )
        if independent_encoders:
            self.encoder = torch.nn.ModuleDict(
                {
                    "treatment": encoder_config.make_module("mlp"),
                    "outcome": encoder_config.make_module("mlp"),
                }
            )
        else:
            self.encoder = encoder_config.make_module("mlp")

        # make outcome model
        outcome_config = layers.ModuleConfig(
            # layers.LayerConfig(hidden_dim, hidden_dim, True, **lkwargs),
            layers.LayerConfig(hidden_dim, 1, False, act=None),
        )
        self.outcome = OutcomeHead(
            outcome_type=outcome_type,
            config=outcome_config,
            vc_spline_degree=outcome_spline_degree,
            vc_spline_knots=outcome_spline_knots,
            glm_family=glm_family,
        )

        # make ratio model
        self.ratio = RatioHead(
            shift_values=shift_values,
            ratio_loss=ratio_loss,
            shift=shift,
            in_dim=hidden_dim,
            ratio_grid_size=ratio_grid_size,
            ratio_spline_degree=ratio_spline_degree,
            ratio_spline_knots=ratio_spline_knots,
            label_smoothing=ratio_label_smoothing,
        )

        # make fluctuation model
        if tr_param_type == "discrete":
            self.tr_model = nn.Parameter(torch.zeros(len(shift_values)))
        elif tr_param_type == "splines":
            self.tr_model = layers.SplineFluctuation(tr_spline_degree, tr_spline_knots)
        elif tr_param_type == "erf":
            self.tr_model = layers.SplineFluctuation(tr_spline_degree, tr_spline_knots)

        # holders for some of the estimators of SRFs
        self.estimator_names = [
            "srf_tr",
            "srf_outcome",
            "srf_ipw",
            "srf_aipw",
            "srf_tr_aipw",
        ]
        self.estimators_batches = defaultdict(list)  # remember to clear on epoch end
        for part in ["train", "val"]:
            for name in self.estimator_names:
                self.register_buffer(f"{name}_{part}", torch.zeros(len(shift_values)))
            self.register_buffer(
                f"srf_estimator_{part}", torch.zeros(len(shift_values))
            )

        # freeze models if necessary
        if self.outcome_freeze:
            for param in self.outcome.parameters():
                param.requires_grad_(False)
            if self.independent_encoders:
                for param in self.encoder["outcome"].parameters():
                    param.requires_grad_(False)

        if self.ratio_freeze:
            for param in self.ratio.parameters():
                param.requires_grad_(False)
            if self.independent_encoders:
                for param in self.encoder["treatment"].parameters():
                    param.requires_grad_(False)

        if not self.tr or self.tr_tmle:
            if self.tr_param_type == "discrete":
                self.tr_model.requires_grad_(False)
            elif self.tr_param_type in ("splines", "erf"):
                for param in self.tr_model.parameters():
                    param.requires_grad_(False)

        self.automatic_optimization = False
        self.finetuning = False

    def fluct_param(self, treatment: Tensor | None = None) -> Tensor:
        if self.tr_param_type == "discrete":
            eps = self.tr_model
        elif self.tr_param_type == "splines":
            eps = self.tr_model(self.shift_values)
        elif self.tr_param_type == "erf":
            eps = self.tr_model(treatment)
        return eps

    def losses_and_estimators(
        self, confounders: Tensor, treatment: Tensor, outcome: Tensor
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        losses = {}
        estimators = {}

        # hidden features
        if self.independent_encoders:
            features_out = self.encoder["outcome"](confounders)
            features_treat = self.encoder["treatment"](confounders)
        else:
            features = self.encoder(confounders)
            features_out = features
            features_treat = features

        # 1. outcome loss
        losses["outcome"], errors = self.outcome.loss(
            treatment,
            features_out,
            outcome,
            return_errors=True,
            noise=self.outcome_training_noise,
        )
        self.lp = self.outcome.lp
        losses["mean_error"] = errors.mean()
        if self.outcome_freeze:
            losses["outcome"] = losses["outcome"].detach()

        # 2. ratio loss
        losses["ratio"] = self.ratio.loss(treatment, features_treat)
        if self.ratio_freeze:
            losses["ratio"] = losses["ratio"].detach()

        # combine ratio and outcome
        # losses["ratio_outcome"] = losses["ratio"] + losses["outcome"]

        # 3. tr loss
        fluct = self.fluct_param(treatment).unsqueeze(0)
        losses["fluct"] = torch.abs(fluct).mean()

        logratio = self.ratio(treatment, features_treat)
        w = torch.exp(logratio)  # density ratio wts
        losses["ratio_norm"] = (w.mean(0) - 1).pow(2).mean()
        if self.ratio_norm:
            w = w / w.mean(0, keepdim=True)

        # consistency, reweighted predictions should be consistent
        shifted = self.shift(treatment[:, None], self.shift_values[None, :])
        #
        n_shifts = len(self.shift_values)
        link = self.glm_family.link
        invlink = self.glm_family.inverse_link
        pred_obs = link(self.outcome(treatment, features_out)).squeeze(1)
        pred_shifted = torch.cat(
            [
                link(self.outcome(shifted[:, i], features_treat))
                for i in range(n_shifts)
            ],
            axis=1,
        )
        pred_shifted_mean = pred_shifted.mean(0)
        tgt = (w * pred_obs[:, None]).mean(0)
        consistency_loss = (tgt - pred_shifted_mean).pow(2).mean()
        losses["consistency"] = consistency_loss

        self.w = w.detach().clone()

        if self.tr_erf:
            # override w
            inputs = torch.cat([treatment[:, None], features_out], 1)
            ps = self.ratio.ps(inputs)[:, None]
            self.ps_eval = ps
            w_tr = 1.0 / (ps + 1e-3)
            if self.ratio_norm:
                w_tr = w_tr / w.mean(0, keepdim=True)
            fluct = fluct.T  #  in erf, fluct is over batch dim
            loss_wts = w_tr
            if self.tr_clever:
                fluct = w_tr * fluct
        else:
            if self.tr_clever:
                fluct = w * fluct
            loss_wts = w

        losses["tr"], errors = self.outcome.loss(
            treatment=treatment,
            features=features_out,
            targets=outcome,
            bias=fluct,
            weights=loss_wts if not self.tr_clever else None,
            detach_intercept=True,
            return_errors=True,
            noise=self.outcome_training_noise,
        )
        if not self.tr_erf:
            losses["tr_mean_error"] = (w * errors).mean()
        else:
            with torch.no_grad():
                losses["tr_mean_error"] = (w_tr * errors).mean()

        if not self.tr:
            # do not comment this. It's needed so that the tarnet
            # does not learn from the tr loss
            losses["tr"] = losses["tr"].detach()

        # 4. estimators per batch
        with torch.no_grad():
            srf_tr = torch.zeros_like(self.shift_values)
            srf_outcome = torch.zeros_like(self.shift_values)
            srf_ipw = torch.zeros_like(self.shift_values)
            srf_aipw = torch.zeros_like(self.shift_values)
            srf_tr_aipw = torch.zeros_like(self.shift_values)

            if self.tr_erf:
                srf_adj = fluct.repeat(1, n_shifts)
            else:
                srf_adj = fluct

            if self.ratio.ratio_loss in ("ps", "hybrid"):
                inputs = torch.cat([treatment[:, None], features_out], 1)
                ps = self.ratio.ps(inputs)

            for i in range(len(self.shift_values)):
                w_i = w[:, i] / w[:, i].mean()
                srf_ipw[i] = (w_i * outcome).mean()
                srf_outcome[i] = pred_shifted_mean[i]
                srf_tr[i] = link(invlink(srf_outcome[i]) + srf_adj[:, i]).mean()
                shift_error_i = (w_i * (outcome - pred_obs)).mean()
                srf_aipw[i] = shift_error_i + srf_outcome[i]
                srf_tr_aipw[i] = shift_error_i + srf_tr[i]

            estimators["srf_tr_aipw"] = srf_tr_aipw
            estimators["srf_aipw"] = srf_aipw
            estimators["srf_ipw"] = srf_ipw
            estimators["srf_outcome"] = srf_outcome
            estimators["srf_tr"] = srf_tr

        # check if ther eis any nan in any of the losses
        for k, v in losses.items():
            if torch.isnan(v).any():
                LOGGER.error(f"NaN in loss {k}")
                raise Exception(f"NaN in loss {k}")

        return losses, estimators

    def training_step(self, batch: tuple[Tensor], _):
        treatment, confounders, outcome = batch
        opt, opt_tr = self.optimizers()

        # 1 -- optimize network (not fluctuation) --
        opt.zero_grad()

        # if finetuning, remove some of the data
        # if not self.finetuning:
        #     k = int(1 / (self.finetune_mask_ratio + 1e-12))
        #     m = confounders.shape[0]
        #     if m > k:
        #         ixs = [i for i in range(m) if i % k != 0]
        #         confounders = confounders[ixs]
        #         treatment = treatment[ixs]
        #         outcome = outcome[ixs]

        losses, estimators = self.losses_and_estimators(confounders, treatment, outcome)

        # total loss
        loss = (
            self.outcome_loss_weight * losses["outcome"]
            + self.ratio_loss_weight * losses["ratio"]
            + self.ratio_norm_loss_weight * losses["ratio_norm"]
            + self.tr_loss_weight * losses["tr"]
            + self.tr_consistency_weight * losses["consistency"]
        )

        # optimize
        # if not self.finetuning or not self.finetune_freeze_nuisance:
        if not (self.tr_tmle and self.finetuning):
            self.manual_backward(loss)
            torch.nn.utils.clip_grad_value_(self.parameters(), self.grad_clip)
            opt.step()
        else:
            loss = loss.detach()

        # save estimators of batch
        for k, v in estimators.items():
            self.estimators_batches[k + "_train"].append(v)

        # 2 -- optimize fluctuation --
        def closure():
            opt_tr.zero_grad()
            lp = self.lp.detach().clone()
            if self.tr_param_type != "erf":
                w = self.w.detach().clone()
                fluct = self.fluct_param(treatment).unsqueeze(0)
            else:
                ps = self.ps_eval.detach().clone()
                w = 1 / (ps + 1e-3)
                w = w / w.mean()
                fluct = self.fluct_param(treatment).unsqueeze(1)

            if self.tr_clever:
                adj = lp + w * fluct
                loss_tr = self.glm_family.loss(adj, outcome[:, None]).mean()
            else:
                adj = lp + fluct
                loss_tr = (w * self.glm_family.loss(adj, outcome[:, None])).mean()
            self.manual_backward(loss_tr)
            if self.tr_param_type == "discrete":
                torch.nn.utils.clip_grad_value_(self.tr_model, self.grad_clip)
            else:
                torch.nn.utils.clip_grad_value_(
                    self.tr_model.parameters(), self.grad_clip
                )
            return loss_tr

        for _ in range(self.tr_freq):
            if self.tr:
                if not self.tr_tmle or self.finetuning:
                    opt_tr.step(closure=closure)

        # log losses and return
        log_dict = {"train/" + k: v for k, v in losses.items()}
        self.log_dict(log_dict, prog_bar=True, on_epoch=True, on_step=False)

        # return loss

    def on_train_epoch_end(self):
        # get num epochs
        ep = self.current_epoch
        if ep >= self.finetune_after:
            self.finetuning = True
            # freeze ratio
            if self.tr_tmle or self.finetune_freeze_nuisance:
                for param in self.encoder.parameters():
                    param.requires_grad_(False)
                    param.grad = None

                for param in self.ratio.parameters():
                    param.requires_grad_(False)
                    param.grad = None

                # freeze outcome head
                for param in self.outcome.parameters():
                    param.requires_grad_(False)
                    param.grad = None

            # unfreeze fluctuation, used in tmle only
            if self.tr_tmle:
                if self.tr_param_type == "discrete":
                    self.tr_model.requires_grad_(True)
                else:
                    for param in self.tr_model.parameters():
                        param.requires_grad_(True)

            # set dropout to 0 in all submodules recursively
            for m in self.modules():
                if isinstance(m, nn.Dropout):
                    m.p = 0.0

            self.tr_freq = 1

        self._on_end("train")

        # step scheduler
        if self.finetune_decrease_lr_after is not None:
            # sch1, sch2 = self.lr_schedulers()
            sch1 = self.lr_schedulers()
            if (
                not self.finetuning
                or not self.finetune_freeze_nuisance
                or not self.tr_tmle
            ):
                sch1.step()
            # sch2.step()

    def validation_step(self, batch: tuple[Tensor], _):
        # set to eval mode
        treatment, confounders, outcome = batch
        losses, estimators = self.losses_and_estimators(confounders, treatment, outcome)

        # save estimators of batch
        for k, v in estimators.items():
            self.estimators_batches[k + "_val"].append(v)

        log_dict = {"val/" + k: float(v) for k, v in losses.items()}
        self.log_dict(log_dict, prog_bar=True, on_epoch=True, on_step=False)

    # function to be applied in both train and validation epoch end
    def _on_end(self, part: Literal["train", "val"]):
        # ground truth
        truth = getattr(self, "true_srf_" + part)
        if truth is None:
            return

        # fetch batches and average esimators
        estimated = {}
        for name in self.estimator_names:
            batches = self.estimators_batches[f"{name}_{part}"]
            estimated[name] = torch.stack(batches).mean(0)
            setattr(self, f"{name}_{part}", estimated[name])

            error = F.mse_loss(estimated[name], truth).pow(0.5)
            self.log(f"{part}/{name}", error)

            # if the estimator specific to the run was declared
            # then save it under a special name srf_estimator
            if self.estimator is not None and (name == f"srf_{self.estimator}"):
                with torch.no_grad():
                    old_val = getattr(self, f"srf_estimator_{part}")
                    diff = estimated[name] - old_val
                    old_val += self.estimator_ma_weight * diff

        ep = self.current_epoch
        if ep == 0 or (ep + 1) % self.plot_every_n_epochs == 0:
            if hasattr(self.logger, "experiment"):
                # plot srf vs truth
                fig, ax = plt.subplots()
                ax.plot(self.shift_values, truth, label="truth", c="black", ls="--")
                for name, value in estimated.items():
                    ax.plot(self.shift_values, value, label=name)
                ax.set_xlabel("shift")
                ax.set_ylabel("srf")
                ax.legend()
                self.logger.experiment.add_figure(f"{part}/fig", fig, ep)

        if self.estimator is not None:
            estimator = getattr(self, f"srf_estimator_{part}")
            error = F.mse_loss(estimator, truth).pow(0.5)
            self.log(f"{part}/estimator_loss", error)

        # log current lr
        lr = self.optimizers()[0].param_groups[0]["lr"]
        self.log(f"lr", lr)

    def on_validation_epoch_end(self):
        self._on_end("val")

    def on_train_epoch_start(self) -> None:
        # clear estimators batches
        self.estimators_batches = defaultdict(list)

    def configure_optimizers(self):
        main_params = list(self.encoder.parameters())
        main_params += list(self.outcome.parameters())
        main_params += list(self.ratio.parameters())
        main_params = [p for p in main_params if p is not self.outcome.intercept]
        param_groups = [dict(params=main_params)]
        param_groups.append(dict(params=[self.outcome.intercept], weight_decay=0.0))

        tr_group = []
        tr_opts = dict(weight_decay=0.0)
        if self.tr_param_type == "discrete":
            tr_group.append(dict(params=[self.tr_model], lr=0.1, **tr_opts))
        elif self.tr_param_type in ("splines", "erf"):
            tr_group.append(dict(params=self.tr_model.parameters(), lr=0.1, **tr_opts))

        # param_groups.append(tr_group[0])

        if self.optimizer == "adam":
            opt = torch.optim.Adam(param_groups, **self.optimizer_opts)
        elif self.optimizer == "sgd":
            opt = torch.optim.SGD(param_groups, **self.optimizer_opts)
            # opt_tr = torch.optim.SGD(tr_group, momentum=self.momentum, lr=0.1)

        opt_tr = torch.optim.Adam(tr_group)
        # opt_tr = torch.optim.LBFGS(tr_group, lr=0.01)
        # self.fake = nn.Parameter(torch.zeros(1), requires_grad=True)
        # opt_tr = torch.optim.Adam([self.fake], lr=0.03)

        # add a scheduler for the optimizer.
        # must decrease by a factor of 10 after reaching finetune_decrease_lr_after
        # and stay at that level there after
        if self.finetune_decrease_lr_after is not None:
            sch1 = torch.optim.lr_scheduler.StepLR(
                opt, step_size=self.finetune_decrease_lr_after, gamma=0.1
            )
            # sch2 = torch.optim.lr_scheduler.StepLR(
            #     opt_tr, step_size=self.finetune_decrease_lr_after, gamma=0.1
            # )
            return [opt, opt_tr], [sch1]
            # return [opt], [sch1]
        else:
            return [opt, opt_tr]
            # return opt
