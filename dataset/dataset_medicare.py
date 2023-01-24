import pandas as pd
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader


def get_iter(data, batch_size, **kwargs):

    treatment, covariates, outcome, offset = (
        data["treatment"],
        data["covariates"],
        data["outcome"],
        data["offset"],
    )
    dataset = TensorDataset(treatment, covariates, outcome, offset)
    iterator = DataLoader(dataset, batch_size=batch_size, **kwargs)
    return iterator


class DataMedicare:
    def __init__(
        self,
        path: str,
        treatment_col: str,
        outcome_col: str,
        offset_col: str,
        categorical_variables: None | list = None,
        columns_to_omit: None | list = None,
        train_prop=0.8,
    ):
        """_summary_

        Args:
            path (str): path to datafile (csv, txt)
            treatment_col (str): column name of treatment variables, e.g pm25
            outcome_column (str): column name of outcome variables, e.g dead
            offset_column (str): column name of offset variables, e.g time_count
            categorical_variables (None | list): One-hot categorical variables among the covariates.
                                                    This is to ensure these variables are not normalized
            columns_to_omit (None | list): columns to drop from the dataset
            train_prop (float, optional): Proportion of training data. Defaults to 0.8.
        """

        if train_prop >= 1 or train_prop <= 0:
            raise ValueError("train_prop must be less than 1 and greater than 0")

        self.data = pd.read_csv(path)
        if columns_to_omit is not None:
            self.data = self.data.drop(columns=columns_to_omit)

        self.categorical_variables = categorical_variables
        self.train_prop = train_prop

        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.offset_col = offset_col

        self.training_set = None
        self.test_set = None

        self.treatment_norm_min = None
        self.treatmnet_norm_max = None

    def init(self):

        # separate data
        treatment, outcome, offset, covariates = self._separate_data()

        # split data
        train_idxs, test_idxs = self.split_data_idx()

        treatment_train, treatment_test = self.split_data(
            treatment, train_idxs, test_idxs
        )
        outcome_train, outcome_test = self.split_data(outcome, train_idxs, test_idxs)
        offset_train, offset_test = self.split_data(offset, train_idxs, test_idxs)

        covariates_train, covariates_test = self.split_data(
            covariates, train_idxs, test_idxs
        )

        covariates_to_standardize = [
            var
            for var in list(covariates_train.columns)
            if var not in self.categorical_variables
        ]

        train_data_to_standardize = covariates_train[covariates_to_standardize]
        mean_covariates = train_data_to_standardize.mean(0)
        std_covariates = train_data_to_standardize.std(0)

        standardized_train_data = (
            train_data_to_standardize - mean_covariates
        ) / std_covariates

        standardized_test_data = (
            covariates_test[covariates_to_standardize] - mean_covariates
        ) / std_covariates

        covariates_train.loc[
            :, list(standardized_train_data.columns)
        ] = standardized_train_data

        covariates_test.loc[
            :, list(standardized_test_data.columns)
        ] = standardized_test_data

        # Clean memory
        del (
            standardized_train_data,
            standardized_test_data,
            train_data_to_standardize,
            mean_covariates,
            std_covariates,
        )

        # get the normalization coeffs

        self.treatment_norm_min = treatment_train.min(0).item()
        self.treatment_norm_max = treatment_train.max(0).item()

        treatment_train, treatment_test = self.min_max_norm(
            treatment_train, self.treatment_norm_min, self.treatment_norm_max
        ), self.min_max_norm(
            treatment_test, self.treatment_norm_min, self.treatment_norm_max
        )

        train_data = {
            "treatment": torch.tensor(treatment_train.values),
            "outcome": torch.tensor(outcome_train.values),
            "covariates": torch.tensor(covariates_train.values),
            "offset": torch.tensor(offset_train.values),
        }
        test_data = {
            "treatment": torch.tensor(treatment_test.values),
            "outcome": torch.tensor(outcome_test.values),
            "covariates": torch.tensor(covariates_test.values),
            "offset": torch.tensor(offset_test.values),
        }

        self.training_set = train_data
        self.test_set = test_data
        # normlize data

        print("The dataset has been initialized")

    def min_max_norm(
        self, data: pd.core.frame.DataFrame, min_val: float, max_val: float
    ) -> pd.core.frame.DataFrame:
        return (data - min_val) / (max_val - min_val)

    def _separate_data(self):

        treatment = self.data[[self.treatment_col]]
        outcome = self.data[[self.outcome_col]]
        offset = self.data[[self.offset_col]]
        covariates = self.data.drop(
            columns=[self.treatment_col, self.outcome_col, self.offset_col]
        )

        return treatment, outcome, offset, covariates

    def split_data(self, data, train_idx, test_idx):

        return data.iloc[train_idx], data.iloc[test_idx]

    def split_data_idx(self):

        train_idxs, test_idxs = self._get_train_test_idx()
        return train_idxs, train_idxs

    def _get_train_test_idx(self):

        n_samples = len(self.data)
        self.n_train = int(n_samples * self.train_prop)

        np.random.seed(5)
        idxs = np.random.permutation(n_samples)

        return idxs[: self.n_train], idxs[self.n_train :]
