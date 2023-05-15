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
        V = torch.randn((num_weights, patients.shape[1]))
        alpha = 2  # dosage bias
        for col in range(V.shape[1]):
            V[:, col] = V[:, col] / torch.linalg.norm(V[:, col], ord=2)

        if variant in (1, 2):
            optimal_dosage = 0.5 * (x @ V[1]) / (x @ V[2])
            dosages = torch.FloatTensor(
                [
                    np.random.beta(alpha, compute_beta(alpha, float(elem)))
                    for elem in optimal_dosage
                ]
            )

        elif variant == 3:
            self._b = 0.75 * (x @ V[1]) / (x @ V[2])
            optimal_dosage = torch.where(self._b >= 0.75, self._b / 3.0, 1.0)
            dosages = torch.FloatTensor(
                [
                    np.random.beta(alpha, compute_beta(alpha, elem))
                    for elem in optimal_dosage
                ]
            )
        dosages = torch.where(optimal_dosage <= 0.001, dosages, 1 - dosages)

        t = torch.tensor(dosages, dtype=torch.float32)
        self._V = torch.FloatTensor(V)
        self._treatment = t
        self._covariates = x

    def linear_predictor(self, covariates: Tensor, treatment: Tensor) -> Tensor:
        V = self._V
        x, t = covariates, treatment

        variant = self.data_opts["dosage_variant"]
        if variant == 1:
            mu = (x @ V[0]) + 12.0 * (x @ V[1]) * t - 12.0 * (x @ V[2]) * (t**2)
        elif variant == 2:
            mu = (x @ V[0]) + torch.sin(torch.pi * ((x @ V[1]) / (x @ V[2])) * t)
        elif variant == 3:
            b = self._b
            mu = (x @ V[0]) + 12.0 * t * (t - b) ** 2

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
