import torch
from torch import nn
from .modules import Encoder, DiscreteDensityEstimator, VCPredictionHead, VCDiscreteEstimator
import numpy


class VCNet(nn.Module):
    def __init__(
        self, encoder_config, num_grids, pred_head_config, spline_degree, spline_knots, dropout=0.0,
    ):
        """Varying coefficient neural networks

        Args:
            encoder_layer_params (Tuple[Tuple]): ((in_dimension, out_dimension, bias), (...))
            num_grids (int): number of grids for linear interpolation in the density estimators

        """

        super(VCNet, self).__init__()

        self.encoder = Encoder(encoder_config, dropout=dropout)
        self.dropout = dropout

        density_estimator_in_dimension = encoder_config[-1][1]

        self.density_estimator = DiscreteDensityEstimator(
            density_estimator_in_dimension, num_grids
        )
        self.prediction_head = VCPredictionHead(
            pred_head_config, spline_degree, spline_knots
        )

    def forward(self, treatment, confounders):
        z = self.encoder(confounders)

        # Density estimator
        probability_score = self.density_estimator(treatment, z)

        # Prediction head
        predicted_outcome = self.prediction_head(treatment, z)
        return {
            "z": z,
            "prob_score": probability_score,
            "predicted_outcome": predicted_outcome.squeeze(1),
        }

    def _initialize_weights(self):
        self.encoder._initialize_weights()
        self.density_estimator._initialize_weights()
        self.prediction_head._initialize_weights()




class RatioNet(nn.Module):
    def __init__(
        self, delta_list, encoder_config, num_grids, pred_head_config, spline_degree_Q, spline_knots_Q, spline_degree_W, spline_knots_W, dropout=0.0,
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

        density_estimator_in_dimension = encoder_config[-1][1]

        self.ratio_estimator = VCDiscreteEstimator(
            density_estimator_in_dimension, num_grids, spline_degree_W, spline_knots_W, a=self.a, b=self.b
        )
        self.prediction_head = VCPredictionHead(
            pred_head_config, spline_degree_Q, spline_knots_W
        )


    def forward(self, treatment, confounders):
        z = self.encoder(confounders)

        # Prediction head
        predicted_outcome = self.prediction_head(treatment, z)
        return {
            "z": z,
            "predicted_outcome": predicted_outcome.squeeze(1),
        }

    def log_ratio(self, treatment, z, delta_ix):
        if isinstance(delta_ix, int):
            delta_ix = torch.full((len(treatment), ), delta_ix, device=treatment.device)
        delta = self.delta_list[delta_ix]

        # Ratio estimator
        return self.ratio_estimator(delta, z)
        

    def _initialize_weights(self):
        self.encoder._initialize_weights()
        self.ratio_estimator._initialize_weights()
        self.prediction_head._initialize_weights()
