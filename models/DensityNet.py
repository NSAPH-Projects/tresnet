import torch
from torch import nn
from modules import Encoder, DensityEstimator
import numpy


class DensityNet(nn.Module):
    def __init__(self, encoder_config, num_grids):
        """Varying coefficient neural networks

        Args:
            encoder_layer_params (Tuple[Tuple]): ((in_dimension, out_dimension, bias), (...))
            num_grids (int): number of grids for linear interpolation in the density estimators

        """

        super(DensityNet, self).__init__()

        self.encoder = Encoder(encoder_config)

        density_estimator_in_dimension = encoder_config[-1][1]

        self.density_estimator = DensityEstimator(
            density_estimator_in_dimension, num_grids
        )

    def forward(self, treatment, confounders):
        z = self.encoder(confounders)

        # Density estimator
        probability_score = self.density_estimator(treatment, z)

        return {
            "z": z,
            "prob_score": probability_score,
        }

    def _initialize_weights(self):
        self.encoder._initialize_weights()
        self.density_estimator._initialize_weights()
