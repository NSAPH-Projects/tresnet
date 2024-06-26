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
        shuffle_batches: bool = False,
        normalize_covariates: bool = True,
        normalize_outcome: bool = True,
        noise_scale: float = 0.1,
        outcome_scale: float = 1.0,  # used only for gaussian
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
        self.outcome_scale = outcome_scale
        self.shuffle_batches = shuffle_batches
        
        self.treatment = None  # load_data must implement
        self.covariates = None  # load_data must implement

        self.load_data()
        self.gen_split_train_val()
        self.generate_outcome_and_counterfactuals()

        # add offset if Poisson case
        if isinstance(self.family, glms.Poisson):
            self.family.off = torch.log(self.outcome.mean())
            # self.family.scale = torch.log(1e-3 + self.outcome).std()


    @abstractmethod
    def linear_predictor(
        self, counterfactual_treatment: Tensor | None = None
    ) -> Tensor:
        """Linear predictor to sample the outcome.

        It takes an optional argument to evaluate the treatment in a counter"""
        pass

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
                m, M = -5, 5
            elif isinstance(self.family, glms.Poisson):
                m, M = 0, 7
            else:
                raise ValueError(f"Unknown family {self.family}")
            lp = m + (M - m) * ((lp - lp_min) / (lp_max - lp_min))
            if isinstance(self.family, glms.Gaussian):
                lp = lp * self.outcome_scale

        generator_seed = random.randint(0, 2**32 - 1)
        sampler = self.family.sample_from_linear_predictor
        self.outcome = sampler(lp, seed=generator_seed, **self.sampler_opts)

        # counterfactuals
        lp_shifted = [self.linear_predictor(covariates, s) for s in shifted]
        if self.normalize_outcome:
            lp_shifted = [m + (M - m) * ((lp - lp_min) / (lp_max - lp_min)) for lp in lp_shifted]
            if isinstance(self.family, glms.Gaussian):
                lp_shifted = [L * self.outcome_scale for L in lp_shifted]

        counterfacutals = [
            sampler(mu, seed=generator_seed, **self.sampler_opts) for mu in lp_shifted
        ]
        self.counterfactuals = torch.stack(counterfacutals, dim=1)

        self.train_srf = self.counterfactuals[self.train_ix].mean(dim=0)
        self.val_srf = self.counterfactuals[self.val_ix].mean(dim=0)
        self.srf = self.counterfactuals.mean(dim=0)

        covariates = self.covariates

        if self.normalize_covariates:
            x = self.covariates[self.train_ix]
            mu, sig = x.mean(dim=0, keepdim=True), x.std(dim=0, keepdim=True)
            sig[sig == 0] = 1
            covariates = (self.covariates - mu) / sig

        self.data_train = dict(
            treatment=self.treatment[self.train_ix],
            shifted_treatment=self.shifted_treatments[:, self.train_ix],
            outcome=self.outcome[self.train_ix],
            covariates=covariates[self.train_ix],
            counterfactuals=self.counterfactuals[self.train_ix, :],
        )
        self.data_val = dict(
            treatment=self.treatment[self.val_ix],
            shifted_treatment=self.shifted_treatments[:, self.val_ix],
            outcome=self.outcome[self.val_ix],
            covariates=covariates[self.val_ix],
            counterfactuals=self.counterfactuals[self.val_ix, :],
        )

    def train_dataloader(self):
        treatment = self.data_train["treatment"]
        covariates = self.data_train["covariates"]
        outcome = self.data_train["outcome"]
        batch_size = (
            self.batch_size if self.batch_size is not None else covariates.shape[0]
        )
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(treatment, covariates, outcome),
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False if self.num_workers == 0 else True,
            shuffle=self.shuffle_batches,
        )

    def val_dataloader(self):
        treatment = self.data_val["treatment"]
        covariates = self.data_val["covariates"]
        outcome = self.data_val["outcome"]
        batch_size = (
            self.batch_size if self.batch_size is not None else covariates.shape[0]
        )
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(treatment, covariates, outcome),
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False if self.num_workers == 0 else True,
            shuffle=False,
        )

    def gen_split_train_val(self) -> None:
        n = self.treatment.shape[0]
        train_ix, test_ix = train_test_split(torch.arange(n), test_size=0.2)
        self.train_ix = train_ix
        self.val_ix = test_ix
