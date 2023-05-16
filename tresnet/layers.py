# parts of this code are modified from https://github.com/lushleaf/varying-coefficient-net-with-functional-tr
# which contains vcnet and drnet implementations.

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor


@dataclass
class LayerConfig:
    """Basic layer configuration"""

    in_dim: int
    out_dim: int
    bias: bool
    act: nn.Module | None = None
    dropout: float | None = None
    is_last: bool = False

    def mlp(self) -> nn.Module:
        return nn.Linear(
            in_features=self.in_dim,
            out_features=self.out_dim,
            bias=self.bias,
        )

    def causal_mlp(self) -> nn.Module:
        return CausalLinear(
            in_features=self.in_dim,
            out_features=self.out_dim,
            bias=self.bias,
            is_last=self.is_last,
        )

    def vc(self, spline_degree: int, spline_knots: int) -> nn.Module:
        return VCLinear(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            spline_degree=spline_degree,
            spline_knots=spline_knots,
            bias=self.bias,
            is_last=self.is_last,
        )

    def drnet(self, is_last: bool = False) -> nn.Module:
        raise NotImplementedError("DR not implemented")

    def make_layer(self, target: str, **kwargs):
        layers = [getattr(self, target)(**kwargs)]

        if self.act is not None:
            layers.append(self.act())

        if self.dropout is not None:
            layers.append(nn.Dropout(self.dropout))

        return nn.Sequential(*layers)


class ModuleConfig:
    def __init__(self, *args: list[LayerConfig]):
        self.config_list = args
        # make sure the last layer hast is_last = True
        for i in range(len(self.config_list) - 1):
            self.config_list[i].is_last = False
        self.config_list[-1].is_last = True

    def make_module(self, target: str, **kwargs):
        layers = [cfg.make_layer(target, **kwargs) for cfg in self.config_list]
        return nn.Sequential(*layers)


class CausalLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_last: bool = False,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features + 1, out_features, bias=bias)
        self.is_last = is_last

    def forward(self, inputs: Tensor) -> Tensor:
        treatment = inputs[:, 0]
        out = self.linear(inputs)
        if not self.is_last:
            out = torch.cat([treatment.unsqueeze(-1), out], dim=1)
        return out


class DiscreteDensityEstimator(nn.Module):
    """Linear layer dividing the support of a probability distribution into a grid
    with n + 1 endpoints and using linear interpolation to estimate the
    probability of the treatment being in each grid cell."""

    def __init__(
        self, in_dim: int, n: int, tmin: float = 0.0, tmax: float = 1.0
    ) -> None:
        super().__init__()
        self.n = n
        self.fc = nn.Linear(in_dim, n + 1, bias=False)
        self.tmin = tmin
        self.tmax = tmax

    def forward(self, inputs: Tensor) -> Tensor:
        treatment = inputs[:, 0]
        features = inputs[:, 1:]
        grid_endpoints = F.softmax(self.fc(features), dim=1)

        # get interpolation params
        a, b, n = self.tmin, self.tmax, self.n
        scaled_treatment = (treatment - a) / (b - a)
        upper_endpoint = torch.ceil(scaled_treatment * n).clamp(0, n)
        lower_endpoint = torch.floor(scaled_treatment * n).clamp(0, n)
        distance_to_lower = scaled_treatment * n - lower_endpoint
        in_support = (scaled_treatment >= 0.0) & (scaled_treatment <= 1.0)

        # obtain  endpoint values and interpolate
        ix = torch.arange(grid_endpoints.shape[0])
        lower_bounds = grid_endpoints[ix, lower_endpoint.long()]
        upper_bounds = grid_endpoints[ix, upper_endpoint.long()]
        prob_score = lower_bounds + (upper_bounds - lower_bounds) * distance_to_lower

        # return probability score, make 0.0 if not in support
        return prob_score * in_support.float()


class VCLinear(nn.Module):
    """Linear Layer where the coefficients are linear functions of the input"""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        spline_degree: int,
        spline_knots: int,
        bias: bool = True,
        is_last: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.spline_degree = spline_degree
        self.spline_knots = spline_knots
        self.bias = bias
        self.is_last = is_last

        # spline basis
        self.sfun = TruncatedPowerBasis(spline_degree, spline_knots)
        self.sdim = self.sfun.basis_dim  # num of basis

        # weights for the W, b weights/biases components of the linear layer
        self.W_wts = nn.Parameter(torch.zeros(self.sdim, self.in_dim * self.out_dim))
        if self.bias:
            self.b_wts = nn.Parameter(torch.zeros(self.sdim, self.out_dim))

        # initialize weights
        nn.init.xavier_uniform_(self.W_wts)
        if self.bias:
            nn.init.xavier_uniform_(self.b_wts)

    def forward(self, inputs: Tensor) -> Tensor:
        treatment = inputs[:, 0]
        features = inputs[:, 1:]

        # compute the basis and multiply by weights
        basis = self.sfun(treatment)  # (batch_size, spline_dim)
        W = (basis @ self.W_wts).view(-1, self.in_dim, self.out_dim)
        out = torch.bmm(features.unsqueeze(1), W).squeeze(1)

        if self.bias:
            bias = basis @ self.b_wts  # (batch_size, out_dim)
            out = out + bias

        if not self.is_last:
            out = torch.cat([treatment.unsqueeze(-1), out], dim=1)

        return out


class DynamicHead(nn.Module):
    def __init__(
        self, config: list[LayerConfig], spline_degree: int, spline_knots: int
    ) -> None:
        """This module generates a varying coefficient prediction head by stacking
        multiple DynamicLinearLayer objects
        """
        super().__init__()
        self.spline_degree = spline_degree
        self.spline_knots = spline_knots

        self.layers = nn.ModuleList()
        for cfg in enumerate(config):
            block = VCLinear(
                cfg.in_dim,
                cfg.out_dim,
                spline_degree,
                spline_knots,
                bias=cfg.bias,
                act=cfg.act,
            )
            self.layers.append(block)

    def forward(self, treatment: Tensor, features: Tensor) -> Tensor:
        x = features
        for layer in self.layers:
            x = layer(treatment, x)
        return x


class TruncatedPowerBasis(nn.Module):
    def __init__(self, degree: int, knots: int) -> None:
        super().__init__()
        self.degree = degree
        self.knots = knots
        self.basis_dim = self.degree + 1 + len(self.knots)

        assert self.degree > 0, "Degree should be greater than 0"
        assert 0 not in knots and 1 not in knots, "Values 0 or 1 cannot be in knots"

    def forward(self, input: Tensor) -> Tensor:
        if len(input.shape) == 2:
            assert input.shape[1] == 1, "Input should be a column vector"
            input = input.squeeze(1)

        deg = self.degree
        out = torch.ones(input.shape[0], self.basis_dim, device=input.device)
        for value in range(1, self.basis_dim):
            if value <= deg:
                out[:, value] = input**value
            else:
                if deg == 1:
                    out[:, value] = F.relu(input - self.knots[value - deg])
                else:
                    out[:, value] = F.relu(input - self.knots[value - deg - 1]) ** deg

        return out


class SplineFluctuation(nn.Module):
    """Model for the targeted regularization fluctuation parameter
    as a spline, similar to as in vcnet"""

    def __init__(
        self, degree: int, knots: int, lower: float = 0.0, upper: float = 1.0
    ) -> None:
        super().__init__()
        self.spline_basis = TruncatedPowerBasis(degree, knots)
        self.bdim = self.spline_basis.basis_dim  # num of basis
        self.weight = nn.Parameter(torch.zeros(self.bdim))
        self.lower = lower
        self.upper = upper

        # init weight
        nn.init.trunc_normal_(self.weight)

    def forward(self, input: Tensor) -> Tensor:
        input = (input - self.lower) / (self.upper - self.lower)
        return self.spline_basis(input) @ self.weight


class Treat_Linear(nn.Module):
    def __init__(self, ind, outd, act="relu", istreat=1, isbias=1, islastlayer=0):
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
            self.treat_weight = nn.Parameter(
                torch.rand(1, self.outd), requires_grad=True
            )
        else:
            self.treat_weight = None

        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "tanh":
            self.act = nn.Tanh()
        elif act == "sigmoid":
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

        self.cfg = cfg  # cfg does NOT include the extra dimension of concat treatment
        self.isenhance = isenhance  # set 1 to concat treatment every layer/ 0: only concat on first layer

        # we default set num of heads = 5
        self.pt = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        self.outdim = -1
        # construct the predicting networks
        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if len(layer_cfg) == 3:
                layer_cfg = (layer_cfg[0], layer_cfg[1], "relu", layer_cfg[2])
            if layer_idx == len(cfg) - 1:  # last layer
                self.outdim = layer_cfg[1]
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                last_layer = Treat_Linear(
                    layer_cfg[0],
                    layer_cfg[1],
                    act=layer_cfg[3],
                    istreat=istreat,
                    isbias=layer_cfg[2],
                    islastlayer=1,
                )
            else:
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                blocks.append(
                    Treat_Linear(
                        layer_cfg[0],
                        layer_cfg[1],
                        act=layer_cfg[3],
                        istreat=istreat,
                        isbias=layer_cfg[2],
                        islastlayer=0,
                    )
                )
        blocks.append(last_layer)
        self.Q1 = nn.Sequential(*blocks)

        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if layer_idx == len(cfg) - 1:  # last layer
                if len(layer_cfg) == 3:
                    layer_cfg = (layer_cfg[0], layer_cfg[1], "relu", layer_cfg[2])
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                last_layer = Treat_Linear(
                    layer_cfg[0],
                    layer_cfg[1],
                    act=layer_cfg[3],
                    istreat=istreat,
                    isbias=layer_cfg[2],
                    islastlayer=1,
                )
            else:
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                blocks.append(
                    Treat_Linear(
                        layer_cfg[0],
                        layer_cfg[1],
                        act=layer_cfg[3],
                        istreat=istreat,
                        isbias=layer_cfg[2],
                        islastlayer=0,
                    )
                )
        blocks.append(last_layer)
        self.Q2 = nn.Sequential(*blocks)

        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if layer_idx == len(cfg) - 1:  # last layer
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                last_layer = Treat_Linear(
                    layer_cfg[0],
                    layer_cfg[1],
                    act=layer_cfg[3],
                    istreat=istreat,
                    isbias=layer_cfg[2],
                    islastlayer=1,
                )
            else:
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                blocks.append(
                    Treat_Linear(
                        layer_cfg[0],
                        layer_cfg[1],
                        act=layer_cfg[3],
                        istreat=istreat,
                        isbias=layer_cfg[2],
                        islastlayer=0,
                    )
                )
        blocks.append(last_layer)
        self.Q3 = nn.Sequential(*blocks)

        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if layer_idx == len(cfg) - 1:  # last layer
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                last_layer = Treat_Linear(
                    layer_cfg[0],
                    layer_cfg[1],
                    act=layer_cfg[3],
                    istreat=istreat,
                    isbias=layer_cfg[2],
                    islastlayer=1,
                )
            else:
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                blocks.append(
                    Treat_Linear(
                        layer_cfg[0],
                        layer_cfg[1],
                        act=layer_cfg[3],
                        istreat=istreat,
                        isbias=layer_cfg[2],
                        islastlayer=0,
                    )
                )
        blocks.append(last_layer)
        self.Q4 = nn.Sequential(*blocks)

        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if layer_idx == len(cfg) - 1:  # last layer
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                last_layer = Treat_Linear(
                    layer_cfg[0],
                    layer_cfg[1],
                    act=layer_cfg[3],
                    istreat=istreat,
                    isbias=layer_cfg[2],
                    islastlayer=1,
                )
            else:
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                blocks.append(
                    Treat_Linear(
                        layer_cfg[0],
                        layer_cfg[1],
                        act=layer_cfg[3],
                        istreat=istreat,
                        isbias=layer_cfg[2],
                        islastlayer=0,
                    )
                )
        blocks.append(last_layer)
        self.Q5 = nn.Sequential(*blocks)

    def forward(self, x):
        # x = [treatment, features]
        out = torch.zeros(x.shape[0], self.outdim).to(x.device)
        t = x[:, 0]

        idx1 = list(
            set(list(torch.where(t >= self.pt[0])[0].numpy()))
            & set(torch.where(t < self.pt[1])[0].numpy())
        )
        idx2 = list(
            set(list(torch.where(t >= self.pt[1])[0].numpy()))
            & set(torch.where(t < self.pt[2])[0].numpy())
        )
        idx3 = list(
            set(list(torch.where(t >= self.pt[2])[0].numpy()))
            & set(torch.where(t < self.pt[3])[0].numpy())
        )
        idx4 = list(
            set(list(torch.where(t >= self.pt[3])[0].numpy()))
            & set(torch.where(t < self.pt[4])[0].numpy())
        )
        idx5 = list(
            set(list(torch.where(t >= self.pt[4])[0].numpy()))
            & set(torch.where(t <= self.pt[5])[0].numpy())
        )

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
