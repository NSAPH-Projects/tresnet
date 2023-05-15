import pandas as pd
import torch
from torch import Tensor

import lightning.pytorch as pl

from tresnet.datamodules.base import TresnetDataModule
from tresnet.datamodules.utils import Min, Max


class IHDP(TresnetDataModule, pl.LightningDataModule):
    def load_data(self):
        x = pd.read_csv(f"data/ihdp/ihdp.csv", usecols=range(2, 27))
        x = torch.FloatTensor(x.values)

        # this indices are used for variuos computations
        self.idx1 = torch.tensor([3, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        self.idx2 = torch.tensor([15, 16, 17, 18, 19, 20, 21, 22, 23, 24])

        # these factors change the behavior, they come from the vcnet paper
        self.factor1 = 1.5
        self.factor2 = 0.5

        # normalize the columns in (0, 1) as in the vcnet paper
        x = (x - x.amin(0, keepdim=True)) / x.amax(0, keepdim=True)

        x1, x2, x3, x4, x5 = [x[:, j] for j in [0, 1, 2, 4, 5]]
        logits = (
            x1 / (1.0 + x2)
            + Max(x3, x4, x5) / (0.2 + Min(x3, x4, x5))
            + ((x[:, self.idx2].mean(1) - x[:, self.idx2].mean()) * 5.0).tanh()
            - 2.0
        )
        logits = logits + self.noise_scale * torch.randn(logits.shape)
        self._treatment = torch.sigmoid(logits)
        self._covariates = x

    def linear_predictor(self, covariates: Tensor, treatment: Tensor) -> Tensor:
        x1, x2, x3, x4, x5 = [self.covariates[:, j] for j in [0, 1, 2, 4, 5]]
        t, x = treatment, covariates
        return (
            torch.sin(t * 3.0 * torch.pi)
            * (
                self.factor1
                * torch.tanh((x[:, self.idx1].mean(1) - x[:, self.idx1].mean()) * 5.0)
                + self.factor2 * torch.exp(0.2 * (x1 - x5)) / (0.1 + Min(x2, x3, x4))
            )
            / (1.2 - t)
        )
