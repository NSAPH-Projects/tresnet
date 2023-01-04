import torch
from torch.utils.data import Dataset, DataLoader


class DatasetFromMatrix(Dataset):
    """Create the pyTorch Dataset object that groes into the dataloader."""

    def __init__(self, data_matrix):
        """
        Args: create a torch dataset from a tensor data_matrix with size n * p
        [treatment, features, outcome]
        """
        self.data_matrix = data_matrix
        self.num_data = data_matrix.shape[0]

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data_matrix[idx, :]

        batch = Batch()

        return {
            "treatment": sample[0],
            "covariates": sample[1:-1],
            "outcome": sample[-1],
        }


def get_iter(data_matrix, batch_size, shuffle=True):
    dataset = DatasetFromMatrix(data_matrix)
    iterator = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return iterator
