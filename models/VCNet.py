import torch
from torch import nn

import numpy


class VCNet(nn.Module):
    def __init__(
        self, encoder_config, num_grids, pred_head_config, spline_degree, spline_knots
    ):
        """Varying coefficient neural networks

        Args:
            encoder_layer_params (Tuple[Tuple]): (in_dimension, out_dimension, bias)
            num_grids (int): number of grids for linear interpolation in the density estimators

        """

        super(VCNet, self).__init__()

        self.encoder = Encoder(encoder_config)

        density_estimator_in_dimension = encoder_config[-1][1]

        self.density_estimator = DensityEstimator(
            density_estimator_in_dimension, num_grids
        )
        self.prediction_head = VCPredictionHead(
            pred_head_config, spline_degree, spline_knots
        )

    def forward(self, treatment, confounders):
        z = self.encoder(confounders)
        probability_score = self.density_estimator(treatment, z)
        predicted_outcome = self.prediction_head(treatment, z)
        return {
            "z": z,
            "prob_score": probability_score,
            "predicted_outcome": predicted_outcome,
        }

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

        layers = []
        for param in layer_params:
            layers.append(
                nn.Linear(in_features=param[0], out_features=param[1], bias=param[2])
            )
            layers.append(nn.ReLU(inplace=True))

        self.encoder = nn.Sequential(*layers)

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


class DynamicLinearLayer(nn.Module):
    def __init__(
        self,
        in_dimension,
        out_dimension,
        is_bias,
        spline_degree,
        spline_knots,
        is_last_layer=0,
    ):
        super(Dynamic_FC, self).__init__()
        self.in_dimension = in_dimension
        self.out_dimension = out_dimension
        self.is_bias = is_bias
        self.spline_degree = spline_knots
        self.spline_knots = spline_knots
        self.is_last_layer = is_last_layer

        self.spline_basis = TruncatedPowerBasis(degree, knots)

        self.num_of_spline_basis = self.spline_basis.num_of_basis  # num of basis

        self.weight = nn.Parameter(
            torch.rand(self.in_dimension, self.out_dimension, self.num_of_spline_basis),
            requires_grad=True,
        )

        if self.is_bias:
            self.bias = nn.Parameter(
                torch.rand(self.out_dimension, self.num_of_spline_basis),
                requires_grad=True,
            )
        else:
            self.bias = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        features = x[:, 1:]
        treatment = x[:, 0]

        hidden = torch.matmul(self.weight.T, features.T).T

        treatment_basis = self.spline_basis(treatment)  # bs, d
        treatment_basis = torch.unsqueeze(treatment_basis, 1)

        out = torch.sum(hidden * treatment_basis, dim=2)  # bs, outd

        if self.isbias:
            out_bias = torch.matmul(self.bias, hidden.T).T
            out = out + out_bias

        out = self.relu(out)

        if not self.is_last_layer:
            # Concatenate to treatment if not last layer
            out = self.relu(out)
            out = torch.cat((torch.unsqueeze(treatment, 1), out), 1)

        return out


class VCPredictionHead(nn.Module):
    def __init__(self, config, spline_degree, spline_knots):

        """This module generates a varying coefficient prediction head by stacking
        multiple DynamicLinearLayer objects
        """
        super(VCPredictionHead, self).__init__()

        self.spline_degree = spline_degree
        self.spline_knots = spline_knots

        blocks = []
        is_last_layer = False
        for idx, params in enumerate(config):
            if idx == len(config) - 1:
                is_last_layer = True

            block = DynamicLinearLayer(
                params[0], params[1], params[2], spline_degree, spline_knots
            )

            blocks.append(block)

        self.prediction_head = nn.Sequential(**block)

    def forward(self, treatment, z):

        treatment_hidden = torch.cat((torch.unsqueeze(treatment, 1), z), 1)
        return self.prediction_head(treatment_hidden)


class TruncatedPowerBasis:
    def __init__(self, degree, knots):

        """
        This class construct the truncated power basis; the data is assumed in [0,1]

        Args:
            degree (int): the degree of truncated basis
            knots (list): the knots of the spline basis; two end points (0,1) should not be included

        """

        self.degree = degree
        self.knots = knots
        self.num_of_basis = self.degree + 1 + len(self.knots)
        self.relu = nn.ReLU(inplace=True)

        if self.degree == 0:
            raise ValueError("Degree should not set to be 0!")

        if not isinstance(self.degree, int):
            raise ValueError("Degree should be int")

        if 0 in knots or 1 in knots:
            raise ValueError("Values 0 or 1 cannot be in knots")

    def __call__(self, x):

        """

        Args:
            x (torch.tensor), treatment (batch size * 1)


        Returns:
            the value of each basis given x; batch_size * self.num_of_basis
        """
        x = x.squeeze()
        out = torch.zeros(x.shape[0], self.num_of_basis)
        for value in range(self.num_of_basis):
            if value <= self.degree:
                if base == 0:
                    out[:, value] = 1.0
                else:
                    out[:, value] = x**value
            else:
                if self.degree == 1:
                    out[:, value] = self.relu(x - self.knots[value - self.degree])
                else:
                    out[:, _] = (
                        self.relu(x - self.knots[value - self.degree - 1])
                    ) ** self.degree

        return out


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
