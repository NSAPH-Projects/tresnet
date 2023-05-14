from dataclasses import dataclass
from typing import Literal
import torch
from torch import nn
import lightning.pytorch as pl

from tresnet.layers import (
    DiscreteDensityEstimator,
    Multi_head,
    Treat_Linear,
    DynamicHead,
)
from tresnet.layers import LayerConfig

@dataclass
class Predictions:
    pred_proba: torch.Tensor | None = None  # predicted logits
    pred_outcome: torch.Tensor | None = None  # predicted outcome
    features: torch.Tensor | None = None  # hidden features


class DynamicNet(nn.Module):
    def __init__(
        self,
        encoder_config: list[LayerConfig],
        outcome_config: list[LayerConfig] | None = None,
        # density_grid_size: int = 0,
        spline_degree: int = 2,
        spline_knots: list[float] = [0.33, 0.66],
        treatment_lower: float = 0.0,
        treatment_upper: float = 1.0,
    ) -> None:
        """Implementation of Varying Coefficient Neural Networks"""
        super().__init__()
        if outcome_config is not None or density_grid_size > 0:
            raise ValueError("Either outcome_config or grid_size must be specified")

        self.lower = treatment_lower
        self.upper = treatment_upper
        self.encoder_config = encoder_config
        self.spline_degree = spline_degree
        self.spline_knots = spline_knots
        self.grid_size = density_grid_size

        # make encoder, density estimator and outcome model (dragone)
        self.encoder = nn.Sequential(*[cfg.make_linear() for cfg in encoder_config])
        hidden_dimension = encoder_config[-1].out_dimension
        if self.grid_size > 0:
            self.density_head = DiscreteDensityEstimator(hidden_dimension, density_grid_size)
        if self.outcome_config is not None:
            self.outcome_head = DynamicHead(outcome_config, spline_degree, spline_knots)

    def forward(
        self, treatment: torch.Tensor, confounders: torch.Tensor
    ) -> Predictions:
        features = self.encoder(confounders)
        normalized_treatment = (treatment - self.upper) / (self.lower - self.upper)

        # Density estimator
        if self.grid_size > 0:
            pred_proba = self.density_head(normalized_treatment, features)
        else:
            pred_proba = None

        # Prediction head
        if self.outcome_config is not None:
            predicted_outcome = self.outcome_head(normalized_treatment, features)

        return Predictions(
            pred_proba=pred_proba,
            pred_outcome=predicted_outcome,
            features=features
        )


class RatioNet(nn.Module):
    def __init__(
        self,
        delta_list,
        encoder_config,
        num_grids,
        pred_head_config,
        spline_degree_Q,
        spline_knots_Q,
        spline_degree_W,
        spline_knots_W,
        dropout=0.0,
        amin=0.0,
        amax=1.0,
    ):
        """Varying coefficient neural networks

        Args:
            encoder_layer_params (Tuple[Tuple]): ((in_dimension, out_dimension, bias), (...))
            num_grids (int): number of grids for linear interpolation in the density estimators

        """

        super().__init__()

        self.encoder = Encoder(encoder_config, dropout=dropout)
        self.dropout = dropout
        self.a, self.b = min(delta_list), max(delta_list)
        self.register_buffer("delta_list", torch.FloatTensor(delta_list))
        self.amin, self.amax = amin, amax

        density_estimator_in_dimension = encoder_config[-1][1]

        self.ratio_estimator = VCDiscreteEstimator(
            density_estimator_in_dimension,
            num_grids,
            spline_degree_W,
            spline_knots_W,
            a=self.a,
            b=self.b,
        )
        self.prediction_head = DynamicHead(
            pred_head_config, spline_degree_Q, spline_knots_W
        )

    def forward(self, treatment, confounders):
        z = self.encoder(confounders)

        # Prediction head
        treatment = (treatment - self.amax) / (self.amin - self.amax)
        predicted_outcome = self.prediction_head(treatment, z)
        return {
            "z": z,
            "predicted_outcome": predicted_outcome.squeeze(1),
        }

    def log_ratio(self, treatment, z, delta_ix):
        if isinstance(delta_ix, int):
            delta_ix = torch.full((len(treatment),), delta_ix, device=treatment.device)
        delta = self.delta_list[delta_ix]

        # Ratio estimator
        return self.ratio_estimator(delta, z)

    def _initialize_weights(self):
        self.encoder._initialize_weights()
        self.ratio_estimator._initialize_weights()
        self.prediction_head._initialize_weights()


class Drnet(nn.Module):
    def __init__(
        self,
        encoder_config,
        num_grids,
        pred_head_config,
        isenhance,
        dropout=0.0,
        amin=0.0,
        amax=1.0,
    ):
        super(Drnet, self).__init__()

        self.encoder = Encoder(encoder_config, dropout=dropout)
        self.dropout = dropout
        self.amin, self.amax = amin, amax

        density_estimator_in_dimension = encoder_config[-1][1]

        self.density_estimator = DiscreteDensityEstimator(
            density_estimator_in_dimension,
            num_grids,
        )
        self.prediction_head = Multi_head(pred_head_config, isenhance)

    def forward(self, treatment, confounders):
        z = self.encoder(confounders)

        # Density estimator
        probability_score = self.density_estimator(treatment, z)

        # Prediction head
        treatment = (treatment - self.amax) / (self.amin - self.amax)
        t_hidden = torch.cat([treatment[:, None], z], 1)
        predicted_outcome = self.prediction_head(t_hidden)
        return {
            "z": z,
            "prob_score": probability_score,
            "predicted_outcome": predicted_outcome.squeeze(1),
        }

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Treat_Linear):
                m.weight.data.normal_(0, 0.01)
                if m.isbias:
                    m.bias.data.zero_()
                if m.istreat:
                    m.treat_weight.data.normal_(
                        0, 1.0
                    )  # this needs to be initialized large to have better performance
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
        self.encoder._initialize_weights()
        self.density_estimator._initialize_weights()
        # self.prediction_head._initialize_weights()


class DensityNet(nn.Module):
    def __init__(self, encoder_config, num_grids, amin=0.0, amax=1.0, dropout=0.0):
        """Density estimation neural networks

        Args:
            encoder_layer_params (Tuple[Tuple]): ((in_dimension, out_dimension, bias), (...))
            num_grids (int): number of grids for linear interpolation in the density estimators

        """

        super(DensityNet, self).__init__()

        self.encoder = Encoder(encoder_config, dropout=dropout)
        self.dropout = dropout
        self.amax, self.amin = amax, amin

        density_estimator_in_dimension = encoder_config[-1][1]

        self.density_estimator = DiscreteDensityEstimator(
            density_estimator_in_dimension,
            num_grids,
        )

    def forward(self, treatment, confounders):
        z = self.encoder(confounders)

        # Density estimator
        treatment = (treatment - self.amax) / (self.amin - self.amax)
        probability_score = self.density_estimator(treatment, z)

        return {
            "z": z,
            "prob_score": probability_score,
        }

    def _initialize_weights(self):
        self.encoder._initialize_weights()
        self.density_estimator._initialize_weights()


class CausalMLP(nn.Module):
    def __init__(
        self,
        encoder_config,
        num_grids,
        pred_head_config,
        dropout=0.0,
        amin=0.0,
        amax=1.0,
    ):
        """Varying coefficient neural networks

        Args:
            encoder_layer_params (Tuple[Tuple]): ((in_dimension, out_dimension, bias), (...))
            num_grids (int): number of grids for linear interpolation in the density estimators

        """

        super().__init__()

        self.encoder = Encoder(encoder_config, dropout=dropout)
        self.dropout = dropout
        self.amin, self.amax = amin, amax

        density_estimator_in_dimension = encoder_config[-1][1]

        self.density_estimator = DiscreteDensityEstimator(
            density_estimator_in_dimension,
            num_grids,
        )
        self.outcome_head = nn.Sequential(
            nn.Linear(
                density_estimator_in_dimension + 1, density_estimator_in_dimension
            ),
            nn.ReLU(),
            nn.Linear(density_estimator_in_dimension, 1),
        )

    def forward(self, treatment, confounders):
        z = self.encoder(confounders)

        # Density estimator
        treatment = (treatment - self.amax) / (self.amin - self.amax)
        probability_score = self.density_estimator(treatment, z)

        # Prediction head
        predicted_outcome = self.outcome_head(
            torch.cat([treatment[:, None], z], axis=1)
        )
        return {
            "z": z,
            "prob_score": probability_score,
            "predicted_outcome": predicted_outcome.squeeze(1),
        }
