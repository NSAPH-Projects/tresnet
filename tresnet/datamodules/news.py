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

        V = torch.rand((6, n_samples, n_features))
        V = V / V.norm(p=2, dim=-1, keepdim=True)

        # alpha = 1 / self.noise_scale
        # tt = 0.5 * torch.mul(V[1], x).sum(1) / torch.mul(V[2], x).sum(1)
        # beta = (alpha - 1) / tt + 2 - alpha
        # beta = np.abs(beta) + 0.0001

        logits = 0.5 * torch.mul(V[1], x).sum(1) / torch.mul(V[2], x).sum(1)
        # standardize logits and add bias
        logits = 0.5 + (logits - logits.mean()) / logits.std()
        m = logits.sigmoid().clamp(min=0.001, max=0.999)
        lam = 1 / self.noise_scale
        alpha = lam * m
        beta = lam * (1 - m)

        # alpha / (alpha + beta
        #  = alpha / ((alpha - 1) / tt + 2)
        #  = alpha * tt / (alpha - 1 + 2 * tt)

        self.treatment = torch.distributions.Beta(alpha, beta).sample()
        self.covariates = x
        self.__V = V
        self.__ifactor = 0.05

    def linear_predictor(self, covariates: Tensor, treatment: Tensor) -> Tensor:
        V = self.__V
        x, t = covariates, treatment

        A = ((torch.mul(V[1], x)).sum(1)) / ((torch.mul(V[2], x)).sum(1))
        B = ((torch.mul(V[4], x)).sum(1)) / ((torch.mul(V[5], x)).sum(1))
        res1a = torch.clamp(torch.exp(0.3 * torch.pi * A - 1), min=-2, max=2)
        res1b = torch.clamp(torch.exp(0.3 * torch.pi * B - 1), min=-2, max=2)
        res2a = self.__ifactor * ((torch.mul(V[0], x)).sum(1))
        res2b = self.__ifactor * ((torch.mul(V[3], x)).sum(1))
        resA = 2 * (4 * (t - 0.5) ** 2 * np.sin(0.5 * torch.pi * t)) * (res1a + res2a)
        resB = 2 * (4 * (t - 0.5) ** 2 * np.sin(0.5 * torch.pi * t)) * (res1b + res2b)
        res = torch.where(t < torch.quantile(t, 0.25), resA, resB)
        return res

