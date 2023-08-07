import pandas as pd
import torch
from torch import Tensor

import lightning.pytorch as pl
from sklearn.neighbors import KernelDensity as KDE
from tresnet.datamodules.base import TresnetDataModule
from tresnet.datamodules.utils import Min, Max


class IHDPB(TresnetDataModule, pl.LightningDataModule):
    def load_data(self):
        """Modification from the IHDP version used in VCNet"""
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
        treatment = torch.sigmoid(logits)

        # quickly density of logits using kernel density estimation
        kde = KDE(kernel="gaussian", bandwidth=0.2).fit(treatment.reshape(-1, 1))
        density = torch.tensor(
            kde.score_samples(treatment.reshape(-1, 1)), dtype=torch.float32
        )

        # eval the kde for every poit in logits
        self.log_density = density
        self.treatment = treatment
        self.covariates = x
        self.lp = self.linear_predictor(x, treatment)

    def linear_predictor(self, covariates: Tensor, treatment: Tensor) -> Tensor:
        t, x = treatment, covariates
        c1 = x[:, self.idx1].mean()
        c2 = x[:, self.idx2].mean()
        z1 = x[:, self.idx1].mean(1) - c1
        z2 = x[:, self.idx2].mean(1) - c2
        d1 = x[:, self.idx1[0]] - x[:, self.idx1[1]]
        d2 = x[:, self.idx2[0]] - x[:, self.idx2[1]]
        m1 = x[:, self.idx1].amin(1)
        m2 = x[:, self.idx2].amin(1)

        f1 = 1.5 * torch.tanh(z1 * 5.0) + 0.5 * torch.exp(0.2 * d1) / (0.1 + m1)
        f2 = 1.5 * torch.tanh(z2 * 5.0) + 0.5 * torch.exp(0.2 * d2) / (0.1 + m2)

        a = torch.quantile(t, 0.25)
        lp = torch.sin(3 * t * torch.pi) / (1.2 - t) * (f1 + f2 * (t < a))
        return lp
