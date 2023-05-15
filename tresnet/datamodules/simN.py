import numpy as np
import torch
from torch import Tensor

import lightning.pytorch as pl 
from tresnet.datamodules.base import TresnetDataModule
from tresnet.datamodules.utils import Min, Max


class SimN(TresnetDataModule, pl.LightningDataModule):
    """This simulation is based on the VCNet paper"""
    def load_data(self):
        n = 700
        x = torch.rand((n, 6))
        x1, x2, x3, x4, x5 = [x[:, j] for j in range(5)]
        logits = (
            (10.0 * Max(x1, x2, x3).sin() + Max(x3, x4, x5).pow(3))
            / (1.0 + (x1 + x5).pow(2))
            + (0.5 * x3).sin() * (1.0 + (x4 - 0.5 * x3).exp())
            + x3.pow(2)
            + 2.0 * x4.sin()
            + 2 * x5
            - 6.5
        )
        t = (logits + self.noise_scale * torch.randn(n)).sigmoid()

        self._treatment = t
        self._covariates = x

    def linear_predictor(self, covariates: Tensor, treatment: Tensor) -> Tensor:
        t, x = treatment, covariates
        x1, x3, x4, x6 = [x[:, j] for j in [0, 2, 3, 5]]
        mu = ((t - 0.5) * 2 * torch.pi).cos() * (
            t**2 + (4 * Max(x1, x6).pow(3)) / (1.0 + 2 * x3.pow(2)) * x4.sin()
        )
        return mu
