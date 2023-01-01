import torch
from torch import nn

import numpy


class Encoder(nn.Module):
    def __init__(self, layer_params):
        """The encoder takes the treatment and covariates and map them to both an
        intermediate vector Z

        Args:
            layer_params (list): _description_
            n_input_features (_type_): _discription_

            n_output_features (_type_): _description_

        """
        super(Encoder, self).__init__()

        # self.n_layers = n_layers
        # self.n_input_features = n_input_features
        # self.n_output_features = n_output_features

        layers = []
        for param in layer_params:
            layers.append(
                nn.Linear(in_features=param[0], out_features=param[1], bias=param[2])
            )
            layers.append(nn.ReLU(inplace=True))

        self.encoder = nn.Sequential(*layers)
        """

        if model_name == 'Vcnet' or model_name == 'Vcnet_tr':
            cfg_density = [(6, 50, 1, 'relu'), (50, 50, 1, 'relu')]
            num_grid = 10
            cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
            degree = 2
            knots = [0.33, 0.66]
            model = Vcnet(cfg_density, num_grid, cfg, degree, knots)
            model._initialize_weights()
        """

    def forward(self, covariates):
        """_summary_

        Args:
            covariates (_type_): confounders

        Returns:
            torch.Tensor : Z
        """

        return self.encoder(covariates)


class DensityEstimator(nn.Module):
    def __init__(self, in_dimension, num_grids, bias=True):
        """This module uses the Encoder and a linear layer + interpolation
        to output intermediate vector z and the conditional density/generalized propensity score

        Args:

            num_grids (_type_): _description_
        """

        super(DensityEstimator, self).__init__()
        self.num_grids = num_grids

        out_dimension = num_grids + 1

        self.fc = nn.Linear(
            in_features=in_dimension, out_features=out_dimension, bias=bias
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, treatment, z):
        """_summary_

        Args:
            z (Tensor): representation vector from the encoder
        """
        out = self.fc(z)
        out = self.softmax(out)

        # x1 = list(torch.arange(0, out.shape[0]))  # List of batch indexes

        # These outputs are needed fto execute the last equation in page 4 of the paper
        # Linear interpolation
        (
            lower_grid_idx,
            upper_grid_idx,
            distance_to_lower,
        ) = get_linear_interpolation_params(treatment, self.num_grids)

        lower_bounds = out[:, lower_grid_idx]  # Get values at the lower grid index

        upper_bounds = out[:, upper_grid_idx]  # Get values at the upper grid index

        gen_propensity_score = (
            lower_bounds + (upper_bounds - lower_bounds) * distance_to_lower
        )

        return gen_propensity_score


class VCNet(nn.Module):
    def __init__(self, encoder_layer_params, num_grids):
        """_summary_

        Args:
            encoder_layer_params (Tuple[Tuple]): (in_dimension, out_dimension, bias)
            num_grids (_type_): _description_

        Returns:
            _type_: _description_
        """

        super(VCNet, self).__init__()

        self.encoder = Encoder(encoder_layer_params)

        density_estimator_in_dimension = encoder_layer_params[-1][1]

        self.density_estimator = DensityEstimator(
            density_estimator_in_dimension, num_grids
        )

    def forward(self, treatment, covariates):
        z = self.encoder(covariates)
        probability_score = self.density_estimator(treatment, z)

        return {"rep_vector": z, "prob_score": probability_score}

    def _initialize_weights(self):
        # TODO: maybe add more distribution for initialization
        for m in self.modules():
            # if isinstance(m, Dynamic_FC):
            #    m.weight.data.normal_(0, 1.0)
            #    if m.isbias:
            #        m.bias.data.zero_()

            # TODO: Ensure parameters defined in the prediction head are initialized as above
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            # elif isinstance(m, DensityEstimator):
            #    m.weight.data.normal_(0, 0.01)
            #    if m.isbias:
            #        m.bias.data.zero_()


def get_linear_interpolation_params(treatment, num_grid):

    """

    Returns:
        Tuple: (upper grid indices, lower grid indices, distance to lower indices)
    """

    upper_grid_idx = torch.ceil(treatment * num_grid)

    distance_to_lower_grid = 1 - (upper_grid_idx - (treatment * num_grid))
    lower_grid_idx = upper_grid_idx - 1

    # This below handles the case when upper bound is zero
    lower_grid_idx += (lower_grid_idx < 0).int()

    return (
        lower_grid_idx.int().tolist(),
        upper_grid_idx.int().tolist(),
        distance_to_lower_grid,
    )
