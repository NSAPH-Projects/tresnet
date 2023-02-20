import torch
from torch import nn

import numpy


class Encoder(nn.Module):
    def __init__(self, layer_params, dropout=0.0):
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
            layers.extend([nn.ReLU(inplace=True), nn.Dropout(dropout)])

        self.encoder = nn.Sequential(*layers)

    def forward(self, covariates):
        """_summary_

        Args:
            covariates (_type_): confounders

        Returns:
            torch.Tensor : Z
        """

        return self.encoder(covariates)

    def _initialize_weights(self):
        # TODO: maybe add more distribution for initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


class DiscreteDensityEstimator(nn.Module):
    def __init__(self, in_dimension, num_grids, bias=True):
        """This module uses the Encoder and a linear layer + interpolation
        to output intermediate vector z and the conditional density/generalized propensity score

        Args:

            num_grids (_type_): _description_
        """

        super(DiscreteDensityEstimator, self).__init__()
        self.num_grids = num_grids

        out_dimension = num_grids + 1

        #! Mauricio Bias is not required when using softmax
        self.fc = nn.Linear(
            in_features=in_dimension, out_features=out_dimension, bias=False # bias
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
        in_support = (treatment >= 0.0) & (treatment <= 1.0)

        ix = torch.arange(out.shape[0])
        lower_bounds = out[ix, lower_grid_idx]  # Get values at the lower grid index
        upper_bounds = out[ix, upper_grid_idx]  # Get values at the upper grid index

        prob_score = lower_bounds + (upper_bounds - lower_bounds) * distance_to_lower

        return prob_score * in_support.float()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


class VCDiscreteEstimator(nn.Module):
    def __init__(self, in_dimension, num_grids, spline_degree, spline_knots, a=0, b=1):
        """This module uses the Encoder and a linear layer + interpolation
        to output intermediate vector z and the conditional density/generalized propensity score

        Args:

            num_grids (_type_): _description_
        """

        super(VCDiscreteEstimator, self).__init__()
        self.num_grids = num_grids
        self.a, self.b = a, b
        out_dimension = num_grids + 1

        #! Mauricio Bias is not required when using softmax
        self.fc = DynamicLinearLayer(
            in_dimension, out_dimension, 1, spline_degree, spline_knots, is_last_layer=True,
        )

    def forward(self, d, z):
        """_summary_

        Args:
            z (Tensor): representation vector from the encoder
        """
        d_hidden = torch.cat([d[:, None], z], 1)
        out = self.fc(d_hidden)

        # x1 = list(torch.arange(0, out.shape[0]))  # List of batch indexes

        # These outputs sare needed fto execute the last equation in page 4 of the paper
        # Linear interpolation
        (
            lower_grid_idx,
            upper_grid_idx,
            distance_to_lower,
        ) = get_linear_interpolation_params(d, self.num_grids, a=self.a, b=self.b)
        in_support = (d >= self.a) & (d <= self.b)
        assert torch.all(in_support), "This layer can't deal with values outside 0-1"

        ix = torch.arange(out.shape[0])
        lower_bounds = out[ix, lower_grid_idx]  # Get values at the lower grid index
        upper_bounds = out[ix, upper_grid_idx]  # Get values at the upper grid index

        prob_score = lower_bounds + (upper_bounds - lower_bounds) * distance_to_lower
        if torch.any(~ in_support):
            raise Exception("")
        return prob_score * in_support.float()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


class DiscreteDensityEstimator(nn.Module):
    def __init__(self, in_dimension, num_grids, bias=True):
        """This module uses the Encoder and a linear layer + interpolation
        to output intermediate vector z and the conditional density/generalized propensity score

        Args:

            num_grids (_type_): _description_
        """

        super(DiscreteDensityEstimator, self).__init__()
        self.num_grids = num_grids

        out_dimension = num_grids + 1

        #! Mauricio Bias is not required when using softmax
        self.fc = nn.Linear(
            in_features=in_dimension, out_features=out_dimension, bias=False # bias
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
        in_support = (treatment >= 0.0) & (treatment <= 1.0)

        ix = torch.arange(out.shape[0])
        lower_bounds = out[ix, lower_grid_idx]  # Get values at the lower grid index
        upper_bounds = out[ix, upper_grid_idx]  # Get values at the upper grid index

        prob_score = lower_bounds + (upper_bounds - lower_bounds) * distance_to_lower

        return prob_score * in_support.float()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


class DynamicLinearLayer(nn.Module):
    def __init__(
        self,
        in_dimension,
        out_dimension,
        is_bias,
        spline_degree,
        spline_knots,
        is_last_layer,
    ):
        super(DynamicLinearLayer, self).__init__()
        self.in_dimension = in_dimension
        self.out_dimension = out_dimension
        self.is_bias = is_bias
        self.spline_degree = spline_degree
        self.spline_knots = spline_knots
        self.is_last_layer = is_last_layer

        self.spline_basis = TruncatedPowerBasis(spline_degree, spline_knots)

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

        #! Mauricio: the following gives a warning, so I commented and used einsum
        # hidden = torch.matmul(self.weight.T, features.T).T
        hidden = torch.einsum('ab,bcd->acd', features, self.weight)

        treatment_basis = self.spline_basis(treatment)  # bs, d
        treatment_basis_unsqueezed = torch.unsqueeze(treatment_basis, 1)

        out = torch.sum(hidden * treatment_basis_unsqueezed, dim=2)  # bs, outd

        if self.is_bias:
            out_bias = torch.matmul(self.bias, treatment_basis.T).T
            out = out + out_bias

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
                params[0],
                params[1],
                params[2],
                spline_degree,
                spline_knots,
                is_last_layer,
            )

            blocks.append(block)

        self.prediction_head = nn.Sequential(*blocks)

    def forward(self, treatment, z):

        treatment_hidden = torch.cat((torch.unsqueeze(treatment, 1), z), 1)
        return self.prediction_head(treatment_hidden)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, DynamicLinearLayer):
                m.weight.data.normal_(0, 1.0)
                if m.is_bias:
                    m.bias.data.zero_()


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
        out = torch.zeros(x.shape[0], self.num_of_basis, device=x.device)
        for value in range(self.num_of_basis):
            if value <= self.degree:
                if value == 0:
                    out[:, value] = 1.0
                else:
                    out[:, value] = x**value
            else:
                if self.degree == 1:
                    out[:, value] = self.relu(x - self.knots[value - self.degree])
                else:
                    out[:, value] = (
                        self.relu(x - self.knots[value - self.degree - 1])
                    ) ** self.degree

        return out


class TargetedRegularizerCoeff(nn.Module):
    def __init__(self, degree, knots):
        super(TargetedRegularizerCoeff, self).__init__()
        self.spline_basis = TruncatedPowerBasis(degree, knots)
        self.num_basis = self.spline_basis.num_of_basis  # num of basis
        self.weight = nn.Parameter(torch.rand(self.num_basis), requires_grad=True)

    def forward(self, t):
        """

        Args:
            t (torch.tensor): Treatment

        Returns:
            torch.tensor
        """
        out = self.spline_basis(t)
        out = torch.matmul(out, self.weight)
        return out

    def _initialize_weights(self):
        self.weight.data.zero_()


def get_linear_interpolation_params(treatment, num_grid, a=0, b=1):

    """

    Returns:
        Tuple: (upper grid indices, lower grid indices, distance to lower indices)
    """

    # upper_grid_idx = torch.ceil(treatment * num_grid)

    # distance_to_lower_grid = 1 - (upper_grid_idx - (treatment * num_grid))
    # lower_grid_idx = upper_grid_idx - 1

    # # This below handles the case when upper bound is zero
    # lower_grid_idx += (lower_grid_idx < 0).int()

    # mauricio: edits
    tmp = (treatment - a) * (b - a)
    upper_grid_idx = torch.ceil(tmp * num_grid).clamp(0, num_grid - 1)
    lower_grid_idx = torch.floor(tmp * num_grid).clamp(0, num_grid - 1)
    distance_to_lower_grid = tmp * num_grid - lower_grid_idx

    return (
        lower_grid_idx.int().tolist(),
        upper_grid_idx.int().tolist(),
        distance_to_lower_grid,
    )


class Treat_Linear(nn.Module):
    def __init__(self, ind, outd, act='relu', istreat=1, isbias=1, islastlayer=0):
        super(Treat_Linear, self).__init__()
        # ind does NOT include the extra concat treatment
        self.ind = ind
        self.outd = outd
        self.isbias = isbias
        self.istreat = istreat
        self.islastlayer = islastlayer

        self.weight = nn.Parameter(torch.rand(self.ind, self.outd), requires_grad=True)

        if self.isbias:
            self.bias = nn.Parameter(torch.rand(self.outd), requires_grad=True)
        else:
            self.bias = None

        if self.istreat:
            self.treat_weight = nn.Parameter(torch.rand(1, self.outd), requires_grad=True)
        else:
            self.treat_weight = None

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = None

    def forward(self, x):
        # x: batch_size * (treatment, other feature)
        x_feature = x[:, 1:]
        x_treat = x[:, [0]]

        out = torch.matmul(x_feature, self.weight)

        if self.istreat:
            out = out + torch.matmul(x_treat, self.treat_weight)
        if self.isbias:
            out = out + self.bias

        if self.act is not None:
            out = self.act(out)

        if not self.islastlayer:
            out = torch.cat((x_treat, out), 1)

        return out

class Multi_head(nn.Module):
    def __init__(self, cfg, isenhance):
        super(Multi_head, self).__init__()

        self.cfg = cfg # cfg does NOT include the extra dimension of concat treatment
        self.isenhance = isenhance  # set 1 to concat treatment every layer/ 0: only concat on first layer

        # we default set num of heads = 5
        self.pt = [0.0, 0.2, 0.4, 0.6, 0.8, 1.]

        self.outdim = -1
        # construct the predicting networks
        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if len(layer_cfg) == 3:
                layer_cfg = (layer_cfg[0], layer_cfg[1], 'relu', layer_cfg[2])
            if layer_idx == len(cfg) - 1:  # last layer
                self.outdim = layer_cfg[1]
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                last_layer = Treat_Linear(layer_cfg[0], layer_cfg[1], act=layer_cfg[3], istreat=istreat, isbias=layer_cfg[2],
                                          islastlayer=1)
            else:
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                blocks.append(Treat_Linear(layer_cfg[0], layer_cfg[1], act=layer_cfg[3], istreat=istreat, isbias=layer_cfg[2],
                                           islastlayer=0))
        blocks.append(last_layer)
        self.Q1 = nn.Sequential(*blocks)

        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if layer_idx == len(cfg) - 1:  # last layer
                if len(layer_cfg) == 3:
                    layer_cfg = (layer_cfg[0], layer_cfg[1], 'relu', layer_cfg[2])
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                last_layer = Treat_Linear(layer_cfg[0], layer_cfg[1], act=layer_cfg[3], istreat=istreat,
                                          isbias=layer_cfg[2],
                                          islastlayer=1)
            else:
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                blocks.append(
                    Treat_Linear(layer_cfg[0], layer_cfg[1], act=layer_cfg[3], istreat=istreat, isbias=layer_cfg[2],
                                 islastlayer=0))
        blocks.append(last_layer)
        self.Q2 = nn.Sequential(*blocks)

        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if layer_idx == len(cfg) - 1:  # last layer
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                last_layer = Treat_Linear(layer_cfg[0], layer_cfg[1], act=layer_cfg[3], istreat=istreat,
                                          isbias=layer_cfg[2],
                                          islastlayer=1)
            else:
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                blocks.append(
                    Treat_Linear(layer_cfg[0], layer_cfg[1], act=layer_cfg[3], istreat=istreat, isbias=layer_cfg[2],
                                 islastlayer=0))
        blocks.append(last_layer)
        self.Q3 = nn.Sequential(*blocks)

        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if layer_idx == len(cfg) - 1:  # last layer
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                last_layer = Treat_Linear(layer_cfg[0], layer_cfg[1], act=layer_cfg[3], istreat=istreat,
                                          isbias=layer_cfg[2],
                                          islastlayer=1)
            else:
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                blocks.append(
                    Treat_Linear(layer_cfg[0], layer_cfg[1], act=layer_cfg[3], istreat=istreat, isbias=layer_cfg[2],
                                 islastlayer=0))
        blocks.append(last_layer)
        self.Q4 = nn.Sequential(*blocks)

        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if layer_idx == len(cfg) - 1:  # last layer
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                last_layer = Treat_Linear(layer_cfg[0], layer_cfg[1], act=layer_cfg[3], istreat=istreat,
                                          isbias=layer_cfg[2],
                                          islastlayer=1)
            else:
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                blocks.append(
                    Treat_Linear(layer_cfg[0], layer_cfg[1], act=layer_cfg[3], istreat=istreat, isbias=layer_cfg[2],
                                 islastlayer=0))
        blocks.append(last_layer)
        self.Q5 = nn.Sequential(*blocks)

    def forward(self, x):
        # x = [treatment, features]
        out = torch.zeros(x.shape[0], self.outdim).to(x.device)
        t = x[:, 0]

        idx1 = list(set(list(torch.where(t >= self.pt[0])[0].numpy())) & set(torch.where(t < self.pt[1])[0].numpy()))
        idx2 = list(set(list(torch.where(t >= self.pt[1])[0].numpy())) & set(torch.where(t < self.pt[2])[0].numpy()))
        idx3 = list(set(list(torch.where(t >= self.pt[2])[0].numpy())) & set(torch.where(t < self.pt[3])[0].numpy()))
        idx4 = list(set(list(torch.where(t >= self.pt[3])[0].numpy())) & set(torch.where(t < self.pt[4])[0].numpy()))
        idx5 = list(set(list(torch.where(t >= self.pt[4])[0].numpy())) & set(torch.where(t <= self.pt[5])[0].numpy()))

        if idx1:
            out1 = self.Q1(x[idx1, :])
            out[idx1, :] = out[idx1, :] + out1

        if idx2:
            out2 = self.Q2(x[idx2, :])
            out[idx2, :] = out[idx2, :] + out2

        if idx3:
            out3 = self.Q3(x[idx3, :])
            out[idx3, :] = out[idx3, :] + out3

        if idx4:
            out4 = self.Q4(x[idx4, :])
            out[idx4, :] = out[idx4, :] + out4

        if idx5:
            out5 = self.Q5(x[idx5, :])
            out[idx5, :] = out[idx5, :] + out5

        return out


