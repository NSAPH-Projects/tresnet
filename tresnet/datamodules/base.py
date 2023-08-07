from abc import abstractmethod
import random
from typing import Callable, Any

from torch import Tensor
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
import torch

from tresnet.shifts import Shift
from tresnet import glms


class TresnetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        shift_values: Tensor,
        shift: Shift,
        family: glms.GLMFamily = glms.Gaussian(),
        batch_size: int | None = None,
        num_workers: int | None = None,
        normalize_covariates: bool = True,
        normalize_outcome: bool = True,
        noise_scale: float = 0.5,
        data_opts: dict = {},
        sampler_opts: dict = {},
    ) -> None:
        super().__init__()
        self.family = family
        self.sampler_opts = sampler_opts
        self.data_opts = data_opts  # not used but accessible in prepare_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shift = shift
        self.shift_values = shift_values
        self.normalize_covariates = normalize_covariates
        self.normalize_outcome = normalize_outcome
        self.noise_scale = noise_scale
        self.load_data()
        self.train_ix, self.val_ix = self.split_train_val()
        self.generate_outcome_and_counterfactuals()

    @abstractmethod
    def linear_predictor(
        self, counterfactual_treatment: Tensor | None = None
    ) -> Tensor:
        """Linear predictor to sample the outcome.

        It takes an optional argument to evaluate the treatment in a counter"""
        pass

    @property
    def treatment(self) -> Tensor:
        return self._treatment

    @property
    def covariates(self) -> Tensor:
        return self._covariates

    @abstractmethod
    def load_data(self):
        pass

    def generate_outcome_and_counterfactuals(self):
        t = self.treatment
        covariates = self.covariates

        # make all shifts
        shifted = [self.shift(t, s) for s in self.shift_values]
        self.shifted_treatments = torch.stack(shifted)

        # compute linear predictors and sample outcome
        # generated_seed will be used for counterfactuals to ensure
        # exogenous noise is the same in a counterfactual curve
        lp = self.linear_predictor(covariates, t)
        if self.normalize_outcome:
            lp_min, lp_max = lp.min(), lp.max()
            if isinstance(self.family, glms.Bernoulli):
                m, M = -10, 10
            elif isinstance(self.family, glms.Gaussian):
                m, M = -100, 100
            elif isinstance(self.family, glms.Poisson):
                m, M = 0, 10
            else:
                raise ValueError(f"Unknown family {self.family}")
            lp = m + (M - m) * ((lp - lp_min) / (lp_max - lp_min))

        generator_seed = random.randint(0, 2**32 - 1)
        sampler = self.family.sample_from_linear_predictor
        outcome = sampler(lp, seed=generator_seed, **self.sampler_opts)

        # counterfactuals
        lp_shifted = [self.linear_predictor(covariates, s) for s in shifted]
        if self.normalize_outcome:
            lp_shifted = [
                m + (M - m) * ((lp - lp_min) / (lp_max - lp_min)) for lp in lp_shifted
            ]

        counterfacutals = [
            sampler(mu, seed=generator_seed, **self.sampler_opts) for mu in lp_shifted
        ]
        counterfactuals = torch.stack(counterfacutals, dim=1)

        self.train_srf = counterfactuals[self.train_ix].mean(dim=0)
        self.val_srf = counterfactuals[self.val_ix].mean(dim=0)
        self.srf = counterfactuals.mean(dim=0)

        covariates = self.covariates

        if self.normalize_covariates:
            x = self.covariates[self.train_ix]
            mu, sig = x.mean(dim=0, keepdim=True), x.std(dim=0, keepdim=True)
            sig[sig == 0] = 1
            covariates = (self.covariates - mu) / sig

        self.data_train = dict(
            treatment=self.treatment[self.train_ix],
            shifted_treatment=self.shifted_treatments[:, self.train_ix],
            outcome=outcome[self.train_ix],
            covariates=covariates[self.train_ix],
            counterfactuals=counterfactuals[self.train_ix, :],
        )
        self.data_val = dict(
            treatment=self.treatment[self.val_ix],
            shifted_treatment=self.shifted_treatments[:, self.val_ix],
            outcome=outcome[self.val_ix],
            covariates=covariates[self.val_ix],
            counterfactuals=counterfactuals[self.val_ix, :],
        )

    def train_dataloader(self):
        treatment = self.data_train["treatment"]
        covariates = self.data_train["covariates"]
        outcome = self.data_train["outcome"]
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(treatment, covariates, outcome),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False if self.num_workers == 0 else True,
            shuffle=False,
        )

    def val_dataloader(self):
        treatment = self.data_val["treatment"]
        covariates = self.data_val["covariates"]
        outcome = self.data_val["outcome"]
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(treatment, covariates, outcome),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False if self.num_workers == 0 else True,
            shuffle=False,
        )

    def split_train_val(self) -> tuple[Tensor, Tensor]:
        n = self.treatment.shape[0]
        return train_test_split(torch.arange(n), test_size=0.2)
