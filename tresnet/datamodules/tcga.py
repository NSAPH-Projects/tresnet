# parts of this code are adapted from https://github.com/ioanabica/SCIGAN
import pickle

import numpy as np
import torch
from torch import Tensor

import lightning.pytorch as pl
from tresnet.datamodules.base import TresnetDataModule


class TCGA(TresnetDataModule, pl.LightningDataModule):
    """This simulation is based on the VCNet paper"""

    def load_data(self):
        # 9659 patients with 4000 features describing them each.
        with open("data/tcga/tcga.p", "rb") as f:
            tcga_data = pickle.load(f)
            patients = normalize_data(tcga_data["rnaseq"])
        x = torch.tensor(patients, dtype=torch.float32)

        variant = self.data_opts["dosage_variant"]

        num_weights = 3
        V = torch.rand((num_weights, patients.shape[1]))
        alpha = 2  # dosage bias
        # noise_scale = self.noise_scale
        # alpha_rescaled = alpha / noise_scale
        for col in range(V.shape[1]):
            V[:, col] = V[:, col] / torch.linalg.norm(V[:, col], ord=2)

        if variant in (1, 2):
            optimal_dosage = 0.5 * (x @ V[1]) / (x @ V[2])
            # normalize to interval [0.01, 0.99]
            m, M = optimal_dosage.min(), optimal_dosage.max()
            optimal_dosage = 0.98 * (optimal_dosage - m) / (M - m) + 0.01
        elif variant == 3:
            self._b = 0.75 * (x @ V[1]) / (x @ V[2])
            m, M = self._b.min(), self._b.max()
            self._b = 0.98 * (self._b - m) / (M - m) + 0.01
            optimal_dosage = torch.where(self._b >= 0.75, self._b / 3.0, 0.999)

        # dosages = []
        dosages_mean = []
        for elem in optimal_dosage:
            beta = compute_beta(alpha, float(elem))
            # rescale according to noise
            # beta_rescaled = beta / noise_scale
            # mu = beta_rescaled / (alpha_rescaled + beta_rescaled)
            mu = beta / (alpha + beta)
            dosages_mean.append(mu)
            # dosages_mean.append(beta_rescaled / (alpha_rescaled + beta_rescaled))
            # dosages.append(np.random.beta(alpha_rescaled, beta_rescaled))
        dosages_mean = torch.FloatTensor(dosages_mean)
        logits = torch.logit(dosages_mean.clamp(0.001, 0.999))
        logits = logits + self.noise_scale * torch.randn(logits.shape)
        treatment = torch.sigmoid(logits)

        # dosages = torch.FloatTensor(dosages)
        # dosages = optimal_dosage.clamp(min=0.01, max=0.99)

        self.__V = torch.FloatTensor(V)
        # self.__dosages_mean = torch.FloatTensor(dosages_mean)
        self.treatment = treatment
        self.covariates = x

    def linear_predictor(self, covariates: Tensor, treatment: Tensor) -> Tensor:
        V = self.__V
        x, t = covariates, treatment

        variant = self.data_opts["dosage_variant"]
        if variant == 1:
            mu = (x @ V[0]) + 12.0 * (x @ V[1]) * t - 12.0 * (x @ V[2]) * (t**2)
        elif variant == 2:
            mu = (x @ V[0]) + torch.sin(torch.pi * ((x @ V[1]) / (x @ V[2])) * t)
        elif variant == 3:
            b = self._b
            mu = (x @ V[0]) * 12.0 * t * (t - b) ** 2

        mu = 10 * mu
        return mu


def normalize_data(patient_features):
    x = (patient_features - np.min(patient_features, axis=0)) / (
        np.max(patient_features, axis=0) - np.min(patient_features, axis=0)
    )
    for i in range(x.shape[0]):
        x[i] = x[i] / np.linalg.norm(x[i])
    return x


def compute_beta(alpha, optimal_dosage):
    if optimal_dosage <= 0.001 or optimal_dosage >= 1.0:
        beta = 1.0
    else:
        beta = (alpha - 1.0) / float(optimal_dosage) + (2.0 - alpha)

    return beta
