import torch
from torch import nn
from .modules import Encoder, DiscreteDensityEstimator, Multi_head, Treat_Linear, VCPredictionHead, VCDiscreteEstimator
import numpy


class VCNet(nn.Module):
    def __init__(
        self, encoder_config, num_grids, pred_head_config, spline_degree, spline_knots, dropout=0.0, amin=0.0, amax=1.0
    ):
        """Varying coefficient neural networks

        Args:
            encoder_layer_params (Tuple[Tuple]): ((in_dimension, out_dimension, bias), (...))
            num_grids (int): number of grids for linear interpolation in the density estimators

        """

        super(VCNet, self).__init__()

        self.encoder = Encoder(encoder_config, dropout=dropout)
        self.dropout = dropout
        self.amin, self.amax = amin, amax

        density_estimator_in_dimension = encoder_config[-1][1]

        self.density_estimator = DiscreteDensityEstimator(
            density_estimator_in_dimension, num_grids, 
        )
        self.prediction_head = VCPredictionHead(
            pred_head_config, spline_degree, spline_knots
        )

    def forward(self, treatment, confounders):
        z = self.encoder(confounders)

        # Density estimator
        treatment = (treatment - self.amax) / (self.amin - self.amax)
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
        self, delta_list, encoder_config, num_grids, pred_head_config, spline_degree_Q, spline_knots_Q, spline_degree_W, spline_knots_W, dropout=0.0, amin=0.0, amax=1.0
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
            density_estimator_in_dimension, num_grids, spline_degree_W, spline_knots_W, a=self.a, b=self.b,
        )
        self.prediction_head = VCPredictionHead(
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
            delta_ix = torch.full((len(treatment), ), delta_ix, device=treatment.device)
        delta = self.delta_list[delta_ix]

        # Ratio estimator
        return self.ratio_estimator(delta, z)
        

    def _initialize_weights(self):
        self.encoder._initialize_weights()
        self.ratio_estimator._initialize_weights()
        self.prediction_head._initialize_weights()


class Drnet(nn.Module):
    def __init__(self, encoder_config, num_grids, pred_head_config, isenhance, dropout = 0.0, amin=0.0, amax=1.0):
        super(Drnet, self).__init__()

        self.encoder = Encoder(encoder_config, dropout=dropout)
        self.dropout = dropout
        self.amin, self.amax = amin, amax

        density_estimator_in_dimension = encoder_config[-1][1]

        self.density_estimator = DiscreteDensityEstimator(
            density_estimator_in_dimension, num_grids,
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
                    m.treat_weight.data.normal_(0, 1.)  # this needs to be initialized large to have better performance
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
            density_estimator_in_dimension, num_grids, 
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
        self, encoder_config, num_grids, pred_head_config,  dropout=0.0, amin=0.0, amax=1.0
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
            density_estimator_in_dimension, num_grids,
        )
        self.outcome_head = nn.Sequential(
            nn.Linear(density_estimator_in_dimension + 1, density_estimator_in_dimension),
            nn.ReLU(),
            nn.Linear(density_estimator_in_dimension, 1)
        )

    def forward(self, treatment, confounders):
        z = self.encoder(confounders)

        # Density estimator
        treatment = (treatment - self.amax) / (self.amin - self.amax)
        probability_score = self.density_estimator(treatment, z)

        # Prediction head
        predicted_outcome = self.outcome_head(torch.cat([treatment[:, None], z], axis=1))
        return {
            "z": z,
            "prob_score": probability_score,
            "predicted_outcome": predicted_outcome.squeeze(1),
        }

    def _initialize_weights(self):
        self.encoder._initialize_weights()
        self.density_estimator._initialize_weights()
        # self.outcome_head._initialize_weights()
