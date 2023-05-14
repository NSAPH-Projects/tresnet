from abc import ABC, abstractmethod, abstractproperty
import random
from torch import Tensor

import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
import torch

from tresnet.shifts import Shift
from tresnet.glms import GLMFamily


class SRFDataset(ABC):
    def __init__(self) -> None:
        self.data = self.load_data()
        self.treatment: Tensor = self.data["treatment"]
        self.covariates: Tensor = self.data["covariates"]

    @abstractproperty
    def covariates(self) -> Tensor:
        pass

    @abstractproperty
    def treatment(self) -> Tensor:
        pass

    @property
    def n_shifts(self) -> int:
        return self.shift_values.shape[0]

    @property
    def n(self) -> int:
        return self.covariates.shape[0]

    @property
    def n_covariates(self) -> int:
        return self.covariates.shape[1]

    @property
    def n_treatments(self) -> int:
        return self.treatment.shape[1]

    def srf(
        self, shift: Shift, shift_values: Tensor, treatment: Tensor, covariates: Tensor
    ) -> Tensor:
        pass


class TresnetDataModule(pl.LightningModule):
    def __init__(
        self,
        shift_values: Tensor,
        shift: Shift,
        family: GLMFamily = GLMFamily.Gaussian,
        batch_size: int | None = None,
        num_workers: int | None = None,
        dataset_kwargs: dict = {},
        outcome_kwargs: dict = {},
    ) -> None:
        super().__init__()
        self.family = family
        self.outcome_kwargs = outcome_kwargs
        self.dataset_kwargs = dataset_kwargs  # not used but accessible in prepare_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shift = shift

    def setup(self, stage: str | None = None):
        self.train_ix, self.val_ix = self.split_train_val()
        t = self.treatment

        # make all shifts
        shifted = [self.shift(t, s) for s in self.shift_values]
        self.shifted_treatments = torch.stack(shifted)

        # compute linear predictors and sample outcome
        # generated_seed will be used for counterfactuals to ensure
        # exogenous noise is the same in a counterfactual curve
        lp = self.linear_predictor()
        generator_seed = random.randint(0, 2**32 - 1)
        sampler = self.family.sample_from_linear_predictor
        self.outcome = sampler(lp, seed=generator_seed, **self.outcome_kwargs)

        # compute counterfactuals
        lp_shifted = [self.linear_predictor(s) for s in shifted]
        counterfacutals = [
            sampler(mu, seed=generator_seed, **self.outcome_kwargs) for mu in lp_shifted
        ]
        self.counterfactuals = torch.stack(counterfacutals, dim=1)

        # split into train and val the treatment, shifted_treatments, 
        # coutcome, covariates, and counterfactuals, and shift-response curves (SRFs)
        self.data_train = dict(
            treatment=self.treatment[self.train_ix],
            shifted_treatment=self.shifted_treatments[:, self.train_ix],
            outcome=self.outcome[self.train_ix],
            covariates=self.covariates[self.train_ix],
            counterfactuals=self.counterfactuals[self.train_ix, :],         
        )
        self.srf_train = self.data_train["counterfactuals"].mean(dim=0)
        self.data_val = dict(
            treatment=self.treatment[self.val_ix],
            shifted_treatment=self.shifted_treatments[:, self.val_ix],
            outcome=self.outcome[self.val_ix],
            covariates=self.covariates[self.val_ix],
            counterfactuals=self.counterfactuals[self.val_ix, :],
        )
        self.srf_val = self.data_val["counterfactuals"].mean(dim=0)

    def train_dataloader(self):
        treatment = self.data_train.treatment
        covariates = self.data_train.covariates
        outcome = self.data_train.outcome
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(treatment, covariates, outcome),
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False if self.num_workers == 0 else True,
            shuffle=False,
        )
    
    def val_dataloader(self):
        treatment = self.data_val.treatment
        covariates = self.data_val.covariates
        outcome = self.data_val.outcome
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(treatment, covariates, outcome),
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False if self.num_workers == 0 else True,
            shuffle=False,
        )

    def split_train_val(self) -> tuple[Tensor, Tensor]:
        n = self.treatment.shape[0]
        return train_test_split(torch.arange(n), test_size=0.2)

    @abstractproperty
    def treatment(self) -> Tensor:
        pass

    @abstractproperty
    def counterfactual_treatment(self) -> Tensor:
        pass

    @abstractproperty
    def covariates(self) -> Tensor:
        pass

    @abstractmethod
    def linear_predictor(
        self, counterfactual_treatment: Tensor | None = None
    ) -> Tensor:
        """Linear predictor to sample the outcome.

        It takes an optional argument to evaluate the treatment in a counter"""
        pass
