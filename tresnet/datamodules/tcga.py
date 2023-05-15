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
        variant = self.data_opts["dosage_variant"]

        num_weights = 3
        V = np.random.normal(loc=0.0, scale=1.0, size=(num_weights, patients.shape[1]))
        for col in range(V.shape[1]):
            V[:, col] = V[:, col] / np.linalg.norm(V[:, col], ord=2)

        if variant == 1:
            dosages = generate_dosage_treatment_1(patients, V)  # generate dosages
        else:
            raise NotImplementedError(f"Dosage variant {variant} not implemented")

        x = torch.tensor(patients, dtype=torch.float32)
        t = torch.tensor(dosages, dtype=torch.float32)
        self._V = torch.FloatTensor(V)
        self._treatment = t
        self._covariates = x

    def linear_predictor(self, covariates: Tensor, treatment: Tensor) -> Tensor:
        V = self._V
        x, t = covariates, treatment
        C = 10
        mu = C * ((x @ V[0]) + 12.0 * (x @ V[1]) * t - 12.0 * (x @ V[2]) * (t**2))
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


def generate_dosage_treatment_3(
    x,
    v,
    dosage_selection_bias=2,
    scaling_parameter=10,
):
    # Treatment 3

    b = 0.75 * np.dot(x, v[1]) / (np.dot(x, v[2]))

    optimal_dosage = np.array([elem / 3.0 if elem >= 0.75 else 1.0 for elem in b])

    alpha = dosage_selection_bias

    dosage = np.array(
        [np.random.beta(alpha, compute_beta(alpha, elem)) for elem in optimal_dosage]
    )
    return dosage


def generate_dosage_treatment_1(
    x,
    v,
    dosage_selection_bias=2,
    scaling_parameter=10,
):
    # Treatment 1

    b = 0.75 * np.dot(x, v[1]) / (np.dot(x, v[2]))

    dosage_selection_bias = 2
    optimal_dosage = np.dot(x, v[1]) / (2.0 * np.dot(x, v[2]))
    alpha = dosage_selection_bias
    dosage = np.array(
        [np.random.beta(alpha, compute_beta(alpha, elem)) for elem in optimal_dosage]
    )
    dosage = np.array(
        [1 - d if o <= 0.001 else d for (d, o) in zip(dosage, optimal_dosage)]
    )
    return dosage
