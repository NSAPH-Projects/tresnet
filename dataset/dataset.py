import torch
from torch.utils.data import Dataset, DataLoader
from torch import cat, stack
from torch import Tensor
import pandas as pd
import numpy as np
import random


DATASETS = (
    "sim-N",  # simu1 simulated data in VCNet (Nie et al., 2021)
    "ihdp-N",  # IHDP modification in VCNet (Nie et al., 2021)
    "news-N",  # News modification in VCNet (Nie et al., 2021)
    "sim-B",  # Simulated data in SCIGAN (Bica et al., 2020)
    "news-B",  # News modification in SCIGAN (Bica et al., 2020)
    "tcga-B",  # TCGA modification in SCIGAN (Bica et al., 2020)
    "sim-T",  # Simulated data in E2B (Taha Bahadori et al., 2022)
)


class DatasetFromMatrix(Dataset):
    """Create the pyTorch Dataset object that groes into the dataloader."""

    def __init__(self, data_matrix):
        """
        Args: create a torch dataset from a tensor data_matrix with size n * p
        [treatment, features, outcome]`z
        """
        self.data_matrix = data_matrix
        self.num_data = data_matrix.shape[0]

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data_matrix[idx, :]

        return {
            "treatment": sample[0],
            "covariates": sample[1:-1],
            "outcome": sample[-1],
        }


def get_iter(data_matrix, batch_size, shuffle=True):
    dataset = DatasetFromMatrix(data_matrix)
    iterator = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return iterator


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def load_data(
    dataset: str, n_train: int | None = None, n_test: int | None = None
) -> tuple[int]:
    """n_train, n_test only useful for simulated datasets"""
    if dataset == "sim-N":  # simu1 simulated data in VCNet (Nie et al., 2021)
        n = n_train + n_test
        x = torch.rand((n, 6))
        x1, x2, x3, x4, x5, x6 = [x[:, j] for j in range(6)]
        logits = (
            (10.0 * Max(x1, x2, x3).sin() + Max(x3, x4, x5).pow(3))
            / (1.0 + (x1 + x5).pow(2))
            + (0.5 * x3).sin() * (1.0 + (x4 - 0.5 * x3).exp())
            + x3.pow(2)
            + 2.0 * x4.sin()
            + 2 * x5
            - 6.5
        )
        t = (logits + 0.5 * torch.randn(n)).sigmoid()
        train_ix = torch.arange(0, n_train)
        test_ix = torch.arange(n_train, n_train + n_test)
        D = {"x": x, "t": t, "train_ix": train_ix, "test_ix": test_ix}
        return D
    elif dataset == "ihdp-N":  # IHDP modification in VCNet (Nie et al., 2021)
        x = pd.read_csv("dataset/ihdp/ihdp.csv", usecols=range(2, 27))
        x = torch.FloatTensor(x.to_numpy())
        n = x.shape[0]

        # normalize the data
        # !mauricio: really weird normalization
        for _ in range(x.shape[1]):
            minval = (x[:, _]).min()
            maxval = (x[:, _]).max()
            x[:, _] = (x[:, _] - minval) / maxval
        # x = (x - x.amin(0)) / x.amax(0)

        cate_idx1 = torch.tensor([3, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        cate_idx2 = torch.tensor([15, 16, 17, 18, 19, 20, 21, 22, 23, 24])

        x1, x2, x3, x4, x5 = [x[:, j] for j in [0, 1, 2, 4, 5]]
        logits = (
            x1 / (1.0 + x2)
            + Max(x3, x4, x5) / (0.2 + Min(x3, x4, x5))
            + ((x[:, cate_idx2].mean(1) - x[:, cate_idx1].mean()) * 5.0).tanh()
            - 2.0
        )
        t = (logits + 0.5 * torch.randn(n)).sigmoid()
        train_ix = torch.arange(0, 473)
        test_ix = torch.arange(473, len(t))
        D = {"x": x, "t": t, "train_ix": train_ix, "test_ix": test_ix}
        return D

    elif dataset == "news-N":  # News modification in VCNet (Niet et al., 2021)
        raise NotImplementedError
    elif dataset == "sim-B":  # Simulated data in SCIGAN (Bica et al., 2020)
        raise NotImplementedError
    elif dataset == "news-B":  # News modification in SCIGAN (Bica et al., 2020)
        raise NotImplementedError
    elif dataset == "tcga-B":  # TCGA modification in SCIGAN (Bica et al., 2020)
        raise NotImplementedError
    elif dataset == "sim-T":  # Simulated data in E2B (Taha Bahadori et al., 2022)
        raise NotImplementedError
    else:
        raise ValueError(dataset)


def support(dataset: str) -> str:
    """Returns link and inverse link"""
    if dataset in ("sim-N", "ihdp-N", "news-N"):  # VCNet datasets (Nie et al., 2021)
        return "unit"
    else:
        return "real"


def outcome(t: Tensor, x: Tensor, dataset: str, noise: Tensor | None = None) -> Tensor:
    if dataset == "sim-N":  # simu1 simulated data in VCNet (Nie et al., 2021)
        x1, x3, x4, x6 = [x[:, j] for j in [0, 2, 3, 5]]
        mu = ((t - 0.5) * 2 * torch.pi).cos() * (
            t**2 + (4 * Max(x1, x6).pow(3)) / (1.0 + 2 * x3.pow(2)) * x4.sin()
        )
        if noise is None:
            noise = 0.5 * torch.randn_like(t)
        y = mu + noise
        return y, noise
    elif dataset == "ihdp-N":  # IHDP modification in VCNet (Niet et al., 2021)
        x1, x2, x3, x4, x5 = [x[:, j] for j in [0, 1, 2, 4, 5]]
        factor1, factor2 = 1.5, 0.5
        cate_idx1 = torch.tensor([3, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        mu = (
            1.0
            / (1.2 - t)
            * torch.sin(t * 3.0 * torch.pi)
            * (
                factor1
                * torch.tanh((x[:, cate_idx1].mean(1) - x[:, cate_idx1].mean()) * 5.0)
                + factor2 * torch.exp(0.2 * (x1 - x5)) / (0.1 + Min(x2, x3, x4))
            )
        )
        if noise is None:
            noise = 0.5 * torch.randn_like(t)
        y = mu + noise
        return y, noise
    elif dataset == "news-N":  # News modification in VCNet (Niet et al., 2021)
        raise NotImplementedError
    elif dataset == "sim-B":  # Simulated data in SCIGAN (Bica et al., 2020)
        raise NotImplementedError
    elif dataset == "news-B":  # News modification in SCIGAN (Bica et al., 2020)
        raise NotImplementedError
    elif dataset == "tcga-B":  # TCGA modification in SCIGAN (Bica et al., 2020)
        raise NotImplementedError
    elif dataset == "sim-T":  # Simulated data in E2B (Taha Bahadori et al., 2022)
        raise NotImplementedError
    else:
        raise ValueError(dataset)


def make_dataset(
    dataset: str,
    delta_list: Tensor,
    **kwargs,
) -> dict[Tensor]:
    """
    delta_std is the number of standard deviations to reduce from the treatment
    n_train, n_test only useful for simulated datasets
    akwargs are passed to load_data
    """
    # -- should be same as vcnet code, but vectorized -- #

    D = load_data(dataset, **kwargs)
    x, t, train_ix, test_ix = D["x"], D["t"], D["train_ix"], D["test_ix"]
    y, noise = outcome(t, x, dataset)

    train_matrix = cat([t[train_ix, None], x[train_ix], y[train_ix, None]], dim=1)
    test_matrix = cat([t[test_ix, None], x[test_ix], y[test_ix, None]], dim=1)

    # -- specific to stochastic interventions -- #
    supp = support(dataset)

    if supp == "unit":  # treatment in (0,1)
        delta_scale = None
        shifted_t = [t * (1 - d) for d in delta_list]
        shift_type = "percent"
    elif supp == "real":  # treatment in real line
        delta_scale = t.std()
        shifted_t = [t - delta_scale * d for d in delta_list]
        shift_type = "subtract"
    else:
        raise NotImplementedError

    # make counterfactuals and shift-response functions
    cfs = stack([outcome(tcf, x, dataset, noise=noise)[0] for tcf in shifted_t], 1)

    # average the counterfactuals for value of delta
    srf_train = cfs[train_ix, :].mean(0)
    srf_test = cfs[test_ix, :].mean(0)

    return {
        "train_matrix": train_matrix,
        "test_matrix": test_matrix,
        "srf_train": srf_train,
        "srf_test": srf_test,
        "delta_scale": delta_scale,
        "shift_type": shift_type,
    }


def Max(*args):
    """point wise max of tensors"""
    return stack(list(args), dim=1).amax(dim=1)


def Min(*args):
    """point wise min of tensors"""
    return stack(list(args), dim=1).amin(dim=1)
