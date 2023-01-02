import numpy as np
import torch
from torch import stack, Tensor
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, TensorDataset
import pandas as pd


def make_treatment(x: Tensor, noise: Tensor) -> Tensor:
    """continuous treatment as in VCnet paper"""
    #! note: bug in the vcnet code uses x[:, cate_idx1].mean()
    cate_idx2 = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    xcate2 = x[:, cate_idx2].mean(1) - x[:, cate_idx2].mean()
    x1, x2, x3, x5, x6 = [x[:, j] for j in [0, 1, 2, 4, 5]]
    C0 = x1 / (1.0 + x2)
    C1 = pmax(x3, x5, x6) / (0.2 + pmin(x3, x5, x6))
    C2 = (5.0 * xcate2).tanh()
    Z = (torch.rand_like(noise) < 0.75).float()
    U = (-Z + 0.5 * xcate2).sigmoid()
    # t = (2.0 * (C0 + C1 + C2) - 4.0 + noise).sigmoid()
    # nex 3 lines are new
    U = U / U.max()
    t = (0.1 + 0.8*(2.0 * (C0 + C1 + C2) - 4.0 + noise)).sigmoid() * U
    t = t / t.max()
    return t

def make_outcome(x: Tensor, t: Tensor, noise: Tensor) -> Tensor:
    """outcome as in VCNet paper"""
    cate_idx1 = [3, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    xcate1 = x[:, cate_idx1].mean(1) - x[:, cate_idx1].mean()
    x1, x2, x3, x5, x6 = [x[:, j] for j in [0, 1, 2, 4, 5]]
    C1 = (3.0 * np.pi * t).sin() / (1.2 - t)
    C2 = (5.0 * xcate1).tanh()
    C3 = (0.2 * (x1 - x6)).exp() / (0.5 + 5.0 * pmin(x2, x3, x5))
    y = C1 * (C2 + C3) + noise
    return y


def pmax(*args: Tensor) -> Tensor:
    return stack(args, 1).amax(1)


def pmin(*args: Tensor) -> Tensor:
    return stack(args, 1).amin(1)


class IHDP_C(pl.LightningDataModule):
    """Variant of the IHDP dataset for evaluating counterfactual
    inference under distributional shifts of a continuous exposure.
    The construction of the treatment and outcome follows VCNet"""

    def __init__(
        self,
        ihdp_path: str,
        dlist: list[float],
        tlist: list[float],
        batch_size: int = 32,
        num_workers: int = 0,
        noise_size: float = 0.25,
    ) -> None:
        super().__init__()
        self.ihdp_path = ihdp_path
        self.dlist = dlist
        self.tlist = tlist
        self.loader_kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

        # def setup(self, stage: str) -> None:
        # read dataset and skip index and binary treatment column
        x = pd.read_csv(self.ihdp_path, usecols=range(2, 27)).values
        #! note: bug in the vcnet code, not standardizing well
        # x = (x - x.min(1, keepdims=True)) / x.max(1, keepdims=True)
        x = (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)
        x = torch.FloatTensor(x)
        N = x.shape[0]

        # treatment
        tnoise = noise_size * torch.randn(N)
        t = make_treatment(x, tnoise)

        # outcome
        onoise = noise_size * torch.randn(N)
        y = make_outcome(x, t, onoise)

        # counterfactuals under distributional exposure shifts
        # here we reuse onoise; the vcnet code doesn't do this, but
        # it makes more sense from the point of view of
        # viewing the noise the exogeneous variables of an individual
        cfs_es = stack([make_outcome(x, d * t, onoise) for d in self.dlist], 1)

        # counterfactuals for average dose response curve
        cfs_dose = stack([make_outcome(x, torch.full_like(t, ti), onoise) for ti in self.tlist], 1)

        # split in train/test
        self.full_dataset = TensorDataset(x, t, y, cfs_es, cfs_dose)
        self.train, self.val = random_split(self.full_dataset, [0.8, 0.2])

    def train_dataloader(self):
        return DataLoader(self.train, **self.loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val, **self.loader_kwargs)
