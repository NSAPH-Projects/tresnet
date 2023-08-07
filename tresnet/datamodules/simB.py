import numpy as np
import torch
from torch import Tensor

import lightning.pytorch as pl 
from tresnet.datamodules.base import TresnetDataModule


class SimB(TresnetDataModule, pl.LightningDataModule):
    """This simulation is based on the E2B paper"""
    def load_data(self):
        n_confounders = 5
        n_samples = 1000

        diag = np.ones((n_confounders))
        off_diag = np.full((n_confounders - 1), fill_value=0.2)  # [0]

        # Create cov matrix
        cov_matrix = np.zeros((n_confounders, n_confounders))

        # Make the matrix tridiagonal
        tridiagonal_matrix = (
            cov_matrix + np.diag(diag, 0) + np.diag(off_diag, -1) + np.diag(off_diag, 1)
        )

        # x is the covariates, there are 5 covariates per sample
        x = np.random.multivariate_normal(
            mean=np.zeros((n_confounders,)), cov=tridiagonal_matrix, size=n_samples
        )
        x = torch.FloatTensor(x)

        beta = torch.FloatTensor(np.random.uniform(low=-1, high=1, size=n_confounders))
        mu_t = np.sin(x @ beta)

        # t is the treatment
        t = torch.sigmoid(mu_t + self.noise_scale * torch.randn(n_samples))

        self.__beta = torch.randn(5)
        self.__gams = torch.randn(4)
        self.treatment = t
        self.covariates = x

    def linear_predictor(self, covariates: Tensor, treatment: Tensor) -> Tensor:
        x, t = covariates, treatment
        beta, gams = self.__beta, self.__gams
        mu = hermit_polynomial(torch.logit(t), gams) + x @ beta
        return mu


def hermit_polynomial(treatment, gams):
    # gamma_0, gamma_1, gamma_2, gamma_3 = np.random.normal(size=4)
    return (
        gams[0]
        + (gams[1] * treatment)
        + (gams[2] * (treatment**2 - 1))
        + (gams[3] * (treatment**3 - (3 * treatment)))
    )