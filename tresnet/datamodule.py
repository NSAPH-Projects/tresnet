import lightning.pytorch as pl
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


def split_causal_data_matrix(matrix: Tensor) -> tuple[Tensor, ...]:
    treatment = matrix[:, 0]
    covariates = matrix[:, 1:-1]
    outcome = matrix[:, -1]
    return treatment, covariates, outcome


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_matrix: Tensor,
        test_matrix: Tensor,
        train_batch_size: int,
        test_batch_size: int,
        num_workers: int = 0,
    ):
        super().__init__()
        self.train_matrix = train_matrix
        self.test_matrix = test_matrix
        self.num_workers = num_workers
        self.n_train = self.train_matrix.shape[0]
        self.n_test = self.test_matrix.shape[0]
        self.n_covariates = self.train_matrix.shape[1] - 2
        self.train_batch_size = min(train_batch_size, self.n_train)
        self.test_batch_size = min(test_batch_size, self.n_test)
        self.training_batches_per_epoch = self.n_train // self.train_batch_size

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            TensorDataset(*split_causal_data_matrix(self.train_matrix)),
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False if self.num_workers == 0 else True,
            shuffle=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            TensorDataset(*split_causal_data_matrix(self.test_matrix)),
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False if self.num_workers == 0 else True,
            shuffle=False,
        )
