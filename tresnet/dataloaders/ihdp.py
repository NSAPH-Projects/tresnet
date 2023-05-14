import pandas as pd
import torch
from torch import Tensor

from tresnet.dataloaders.datamodule import TresnetDataModule
from tresnet.dataloaders.utils import Min, Max


class IHDP(TresnetDataModule):
    def prepare_data(self):
        x = pd.read_csv(f"{self.root}/ihdp.csv", usecols=range(2, 27))

        # this indices are used for variuos computations
        self.idx1 = torch.tensor([3, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        self.idx2 = torch.tensor([15, 16, 17, 18, 19, 20, 21, 22, 23, 24])

        # these factors change the behavior, they come from the vcnet paper
        self.factor1 = 1.5
        self.factor2 = 0.5

        # normalize the columns in (0, 1) as in the vcnet paper
        x = (x - x.min(0, keepdim=True)) / x.max(0, keepdim=True)

        x1, x2, x3, x4, x5 = [x[:, j] for j in [0, 1, 2, 4, 5]]
        logits = (
            x1 / (1.0 + x2)
            + Max(x3, x4, x5) / (0.2 + Min(x3, x4, x5))
            + ((x[:, self.idx2].mean(1) - x[:, self.idx1].mean()) * 5.0).tanh()
            - 2.0
        )
        self.info = dict(
            covariates=x,
            treatment=torch.normal(logits, self.noise_scale).sigmoid(),
            treatment_range="unit",
        )

    @property
    def covariates(self) -> Tensor:
        return self.info["covariates"]

    @property
    def treatment(self) -> Tensor:
        return self.info["treatment"]

    @property
    def linear_predictor(self) -> Tensor:
        x1, x2, x3, x4, x5 = [self.covariates[:, j] for j in [0, 1, 2, 4, 5]]
        t, x = self.treatment, self.covariates
        return (
            torch.sin(t * 3.0 * torch.pi)
            * (
                self.factor1
                * torch.tanh((x[:, self.idx1].mean(1) - x[:, self.idx1].mean()) * 5.0)
                + self.factor2 * torch.exp(0.2 * (x1 - x5)) / (0.1 + Min(x2, x3, x4))
            )
            / (1.2 - t)
        )
