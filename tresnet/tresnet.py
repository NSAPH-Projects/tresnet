from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import lightning.pytorch as pl
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

from tresnet import layers, shifts


@dataclass
class TresnetOuputs:
    pred_logratio: Tensor | None = None  # predicted logits
    pred_proba: Tensor | None = None  # predicted logits
    pred_outcome: Tensor | None = None  # predicted outcome
    features: Tensor | None = None  # hidden features
    fluctuation: Tensor | None = None  # fluctuations


def link_and_inverse_link(
    loss_family: Literal["gaussian", "bernoulli", "poisson"]
) -> tuple[callable, callable]:
    if loss_family == "gaussian":
        return lambda x: x, lambda x: x
    elif loss_family == "bernoulli":
        return torch.sigmoid, lambda x: torch.logit(x + 1e-8)
    elif loss_family == "poisson":
        return torch.exp, torch.log


class OutcomeHead(nn.Module):
    """Outcome head model"""

    def __init__(
        self,
        outcome_type: Literal["vc", "mlp", "piecewise"],
        config: layers.ModuleConfig,
        vc_spline_degree: int = 2,
        vc_spline_knots: list[float] = [0.33, 0.66],
        loss_family: Literal["gaussian", "bernoulli", "poisson"] = "gaussian",
    ) -> None:
        super().__init__()
        self.outcome_type = outcome_type
        self.loss_family = loss_family
        if outcome_type == "vc":
            kwargs = dict(spline_degree=vc_spline_degree, spline_knots=vc_spline_knots)
        elif outcome_type == "mlp":
            kwargs = dict(causal=True)  # add dimension
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
        return_mean_error: bool = False,
    ) -> Tensor:
        # obtain predictor
        pred = self(treatment, features, detach_bias=detach_intercept) + bias

        # because pred is either has either one column for the outcom eloss
        # or n=len(shift_values) columns for the targeted regularization
        targets = targets[:, None].repeat(1, pred.shape[1])

        # eval loss per item
        if self.loss_family == "gaussian":
            loss_ = F.mse_loss(pred, targets, reduction="none")
        elif self.loss_family == "bernoulli":
            loss_ = F.binary_cross_entropy_with_logits(pred, targets, reduction="none")
        elif self.loss_family == "poisson":
            loss_ = F.poisson_nll_loss(pred, targets, log_input=True, reduction="none")

        # aggregate
        if weights is not None:
            loss_ *= weights
        loss_ = loss_.mean()

        if not return_mean_error:
            return loss_
        else:
            link, _ = link_and_inverse_link(self.loss_family)
            mean_error = (targets - link(pred)).mean()
            return loss_, mean_error


class RatioHead(nn.Module):
    """Ratio head model"""

    def __init__(
        self,
        shift_values: list[float],
        ratio_type: Literal["ps", "hybrid", "classifier"],
        shift_type: Literal["percent", "subtract", "cutoff"],
        in_dim: int,
        ratio_grid_size: int,
        ratio_spline_degree: int = 2,
        ratio_spline_knots: list[float] = [0.33, 0.66],
        label_smoothing: float = 0.01,
    ) -> None:
        super().__init__()
        self.ratio_type = ratio_type
        self.shift_type = shift_type
        self.register_buffer("shift_values", torch.FloatTensor(shift_values))
        self.label_smoothing = label_smoothing
        self.shift = getattr(shifts, shift_type.capitalize())()

        # validate shift type and ratio type
        if not ratio_type in ("ps", "hybrid"):
            if not self.shift.has_inverse():
                raise ValueError("shift function must have inverse and logdet")

        # ratio model
        if ratio_type in ("ps", "hybrid"):
            self.ps = layers.DiscreteDensityEstimator(in_dim, ratio_grid_size)
        elif ratio_type == "classifier":
            # classifier with num_shifts heads
            args = [in_dim, 1, ratio_spline_degree, ratio_spline_knots]
            self.class_logits = nn.ModuleList(
                [layers.VCLinear(*args) for _ in range(len(self.shift_values))]
            )
        else:
            raise NotImplementedError(f"ratio_type {ratio_type} not implemented")

    def forward(self, treatment: Tensor, features: Tensor) -> Tensor:
        # there's two cases two handle, when treatment is a vector
        # and where treatment is a column, each column has been shifted
        # we want to be smart about broadcasting along shifts
        shift_values = self.shift_values[None, :]
        if len(treatment.shape) == 1:
            treatment = treatment[:, None].repeat(1, len(self.shift_values))

        if self.ratio_type in ("ps", "hybrid"):
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
            numerator = torch.log(ps_inv + 1e-6) + logdet
            denominator = torch.log(ps_obs + 1e-6)
            log_ratio = numerator - denominator

        elif self.ratio_type == "classifier":
            log_ratio = []
            for i in range(len(self.shift_values)):
                inputs = torch.cat([treatment[:, i, None], features], 1)
                log_ratio.append(self.class_logits[i](inputs))
            log_ratio = torch.cat(log_ratio, 1)

        return log_ratio

    def loss(self, treatment: Tensor, features: Tensor) -> Tensor:
        inputs = torch.cat([treatment[:, None], features], 1)
        if self.ratio_type == "ps":
            # likelihood/erm loss
            ps_obs = self.ps(inputs)
            loss_ = -torch.log(ps_obs + 1e-6).mean()

        elif self.ratio_type in ("hybrid", "classifier"):
            # classifier loss, but compute ratio from ps
            shifted = self.shift(treatment[:, None], self.shift_values[None, :])
            ratio1 = self(shifted, features)
            ratio2 = self(treatment, features)
            logits = torch.cat([ratio2, ratio1])
            tgts = torch.cat([torch.zeros_like(ratio2), torch.ones_like(ratio1)])
            tgts = tgts.clamp(self.label_smoothing / 2, 1 - self.label_smoothing / 2)
            loss_ = 0.5 * F.binary_cross_entropy_with_logits(logits, tgts)

        return loss_


class Tresnet(pl.LightningModule):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        shift_values: list[float],
        shift_type: Literal["percent", "subtract", "cutoff"] = "percent",
        outcome_head: bool = True,
        outcome_type: str = Literal["vc", "mlp", "drnet"],
        outcome_spline_degree: int = 2,
        outcome_spline_knots: list[float] = [0.33, 0.66],
        outcome_family: Literal["gaussian", "bernoulli", "poisson"] = "gaussian",
        ratio_head: bool = True,
        ratio_type: Literal["ps", "hybrid", "classifier"] = "ps",
        ratio_spline_degree: int = 2,
        ratio_spline_knots: list[float] = [0.33, 0.66],
        ratio_grid_size: int = 10,
        ratio_label_smoothing: float = 0.01,
        ratio_loss_weight: float = 1.0,
        tr: bool = True,  # targeted regularization
        tr_spline_degree: int = 2,
        tr_spline_knots: list[float] = list(np.linspace(0, 1, num=10)[1:-1]),
        tr_param_type: Literal["discrete", "spline"] = "discrete",
        tr_use_clever: bool = True,
        tr_weight_norm: bool = False,
        tr_loss_weight: float = 0.1,
        act: nn.Module = nn.SiLU,
        opt_lr: float = 1e-3,
        opt_weight_decay: float = 5e-3,
        opt_optimizer: Literal["adam", "sgd"] = "adam",
        dropout: float = 0.0,
        true_train_srf: Tensor | None = None,
        true_val_srf: Tensor | None = None,
        plot_every_n_epochs: int = 100,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.register_buffer("shift_values", torch.FloatTensor(shift_values))
        self.outcome_head = outcome_head
        self.outcome_family = outcome_family
        self.ratio_head = ratio_head
        self.ratio_loss_weight = ratio_loss_weight
        self.tr = tr
        self.tr_param_type = tr_param_type
        self.tr_use_clever = tr_use_clever
        self.tr_weight_norm = tr_weight_norm
        self.tr_loss_weight = tr_loss_weight
        self.optimizer = opt_optimizer
        self.lr = opt_lr
        self.wd = opt_weight_decay
        self.true_train_srf = true_train_srf
        self.true_val_srf = true_val_srf
        self.shift = getattr(shifts, shift_type.capitalize())()
        self.plot_every_n_epochs = plot_every_n_epochs

        # layer kwargs
        lkwargs = dict(
            act=act,
            dropout=dropout,
        )

        # make feature encoder
        encoder_config = layers.ModuleConfig(
            layers.LayerConfig(in_dim, hidden_dim, True, **lkwargs),
            layers.LayerConfig(hidden_dim, hidden_dim, True, **lkwargs),
        )
        self.encoder = encoder_config.make_module("mlp")

        # make outcome model
        if outcome_head:
            outcome_config = layers.ModuleConfig(
                layers.LayerConfig(hidden_dim, hidden_dim, True, **lkwargs),
                layers.LayerConfig(hidden_dim, 1, False, act=None),
            )
            self.outcome = OutcomeHead(
                outcome_type=outcome_type,
                config=outcome_config,
                vc_spline_degree=outcome_spline_degree,
                vc_spline_knots=outcome_spline_knots,
                loss_family=outcome_family,
            )

        # make ratio model
        if ratio_head:
            self.ratio = RatioHead(
                shift_values=shift_values,
                ratio_type=ratio_type,
                shift_type=shift_type,
                in_dim=hidden_dim,
                ratio_grid_size=ratio_grid_size,
                ratio_spline_degree=ratio_spline_degree,
                ratio_spline_knots=ratio_spline_knots,
                label_smoothing=ratio_label_smoothing,
            )

        # make fluctuation model
        if tr:
            if tr_param_type == "discrete":
                self.tr_model = nn.Parameter(torch.zeros(len(shift_values)))
            elif tr_param_type == "spline":
                self.tr_model = layers.SplineFluctuation(
                    tr_spline_degree, tr_spline_knots
                )

        # for each epoch, store the train and val srf
        self.train_srf = []
        self.val_srf = []

    def forward(self, treatment: Tensor, confounders: Tensor) -> TresnetOuputs:
        outputs = {}

        # encode features
        features = self.encoder(confounders)
        outputs["features"] = features

        # outcome model
        if self.outcome_head:
            pred_outcome = self.outcome(treatment, features)
            outputs["pred_outcome"] = pred_outcome

        # ratio model
        if self.ratio_head:
            logratio = self.ratio(treatment, features)
            outputs["pred_logratio"] = logratio

        # fluctuation model
        if self.tr:
            outputs["fluctuation"] = self.fluct_param()

        return TresnetOuputs(**outputs)

    def fluct_param(self) -> Tensor:
        if self.tr_param_type == "discrete":
            eps = self.tr_model
        elif self.tr_param_type == "spline":
            eps = self.tr_model(self.shift_values)
        return eps

    def compute_losses(
        self, confounders: Tensor, treatment: Tensor, outcome: Tensor
    ) -> dict[str, Tensor]:
        losses = defaultdict(lambda: 0.0)

        # hidden features
        features = self.encoder(confounders)

        # 1. outcome loss
        if self.outcome_head:
            losses["outcome"], losses["mean_error"] = self.outcome.loss(
                treatment, features, outcome, return_mean_error=True
            )

        # 2. ratio loss
        if self.ratio_head:
            losses["ratio"] = self.ratio.loss(treatment, features)

        # 3. tr loss
        if self.tr:
            fluct = self.fluct_param().unsqueeze(0)
        else:
            fluct = torch.zeros_like(self.shift_values).unsqueeze(0)

        logratio = self.ratio(treatment, features)
        w = torch.exp(logratio.clamp(-10, 10))  # density ratio wts
        if self.tr_weight_norm:
            w = w / w.mean(0, keepdim=True)

        if self.tr_use_clever:
            fluct = w * fluct
            w = None

        losses["tr"], losses["tr_mean_error"] = self.outcome.loss(
            treatment=treatment,
            features=features,
            targets=outcome,
            bias=fluct,
            weights=w,
            detach_intercept=True,
            return_mean_error=True,
        )

        # 4. estimate counterfactuals under shifts
        with torch.no_grad():
            if not self.tr:
                srf_adj = torch.zeros_like(self.shift_values)
            else:
                srf_adj = fluct.mean(0)
            shifted = self.shift(treatment[:, None], self.shift_values[None, :])
            srf = torch.zeros_like(self.shift_values)
            for i in range(len(self.shift_values)):
                link, _ = link_and_inverse_link(self.outcome_family)
                pred = self.outcome(shifted[:, i], features).squeeze(1)
                srf[i] = link(pred + srf_adj[i]).mean()
            losses["srf"] = srf

        return losses

    def training_step(self, batch: tuple[Tensor], _):
        treatment, confounders, outcome = batch
        losses = self.compute_losses(confounders, treatment, outcome)

        # total loss
        loss = (
            losses["outcome"]
            + self.ratio_loss_weight * losses["ratio"]
            + self.tr_loss_weight * losses["tr"]
        )

        # get srf estimate
        train_srf = losses.pop("srf")
        self.train_srf.append(train_srf)

        # log losses and return
        log_dict = {"train/" + k: float(v) for k, v in losses.items()}
        self.log_dict(log_dict, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch: tuple[Tensor], _):
        treatment, confounders, outcome = batch
        losses = self.compute_losses(confounders, treatment, outcome)

        # get srf estimate
        val_srf = losses.pop("srf")
        self.val_srf.append(val_srf)

        log_dict = {"val/" + k: float(v) for k, v in losses.items()}
        self.log_dict(log_dict, prog_bar=True, on_epoch=True, on_step=False)

    # function to be applied in both train and validation epoch end
    def _on_end_helper(self, what: Literal["train", "val"]):
        buffer = getattr(self, what + "_srf")
        srf = torch.stack(buffer).mean(0)
        truth = getattr(self, "true_" + what + "_srf")

        if truth is not None:
            error = F.mse_loss(srf, truth)
            self.log(what + "/srf_error", error)

            if what == "val":  # for tensorboard
                self.log("hp_metric", error)

            ep = self.current_epoch
            if ep == 0 or (ep + 1) % self.plot_every_n_epochs == 0:
                # make a plot
                fig, ax = plt.subplots()
                ax.plot(self.shift_values, truth, label="true")
                ax.plot(self.shift_values, srf, label="estimated")
                ax.set_xlabel("shift")
                ax.set_ylabel("srf")
                ax.legend()
                self.logger.experiment.add_figure(what + "/srf", fig, ep)

        buffer.clear()

    def on_train_epoch_end(self):
        self._on_end_helper("train")

    def on_validation_epoch_end(self):
        self._on_end_helper("val")

    def configure_optimizers(self):
        main_params = list(self.encoder.parameters())
        if self.outcome_head:
            params_except_intercept = [
                p for p in self.outcome.parameters() if p is not self.outcome.intercept
            ]
            main_params += params_except_intercept
        if self.ratio_head:
            main_params += list(self.ratio.parameters())

        param_groups = [dict(params=main_params, lr=self.lr, weight_decay=self.wd)]

        if self.outcome_head:
            intercept = self.outcome.intercept
            param_groups.append(dict(params=[intercept], weight_decay=0.0))

        if self.tr:
            param_groups.append(dict(params=[self.tr_model], weight_decay=0.0))

        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(param_groups, lr=self.lr)
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(param_groups, momentum=self.momentum)

        return optimizer
