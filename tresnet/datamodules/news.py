import numpy as np
import torch
from torch import Tensor

import lightning.pytorch as pl 
from tresnet.datamodules.base import TresnetDataModule


class News(TresnetDataModule, pl.LightningDataModule):
    def load_data(self):
        # Load preprocessed data from numpy file
        x = np.load("data/news/news_preprocessed.npy")
        x = torch.FloatTensor(x)
        n_samples, n_features = x.shape

        # Normalize the data
        x = x / x.amax(0)

        V = torch.randn((3, n_samples, n_features))
        V = V / V.norm(p=2, dim=-1, keepdim=True)

        alpha = 1 / self.noise_scale
        tt = 0.5 * torch.mul(V[1], x).sum(1) / torch.mul(V[2], x).sum(1)

        betas = (alpha - 1) / tt + 2 - alpha
        betas = np.abs(betas) + 0.0001

        t = torch.distributions.Beta(alpha, betas).sample()

        self._treatment = t
        self._covariates = x
        self._V = V

    def linear_predictor(self, covariates: Tensor, treatment: Tensor) -> Tensor:
        V = self._V
        x, t = covariates, treatment

        A = ((torch.mul(V[1], x)).sum(1)) / ((torch.mul(V[2], x)).sum(1))
        res1 = torch.clamp(torch.exp(0.3 * torch.pi * A - 1), min=-2, max=2)
        res2 = 20.0 * ((torch.mul(V[0], x)).sum(1))
        res = 2 * (4 * (t - 0.5) ** 2 * np.sin(0.5 * torch.pi * t)) * (res1 + res2)

        return res

