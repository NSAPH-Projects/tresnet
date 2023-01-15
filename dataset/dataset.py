import torch
from torch.utils.data import Dataset, DataLoader
from torch import cat, stack
from torch import Tensor
import pandas as pd

DATASETS = ("ihdp-N", "sim-N", "news-N", "sim-B", "news-B", "tcga-B", "sim-T")


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

        return {
            "treatment": sample[0],
            "covariates": sample[1:-1],
            "outcome": sample[-1],
        }


def get_iter(data_matrix, batch_size, shuffle=True):
    dataset = DatasetFromMatrix(data_matrix)
    iterator = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return iterator


def load_data(dataset: str, n_train: int | None, n_test: int | None) -> tuple[int]:
    """n_train, n_test only useful for simulated datasets"""
    if dataset == "sim-N":  # simu1 simulated data in VCNet (Nie et al., 2021)
        n = n_train, n_test
        x1, x2, x3, x4, x5, x6 = [torch.rand(n) for j in range(5)]
        x = stack(x1, x2, x3, x4, x5, x6)
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
    elif dataset == "ihdp-N":  # IHDP modification in VCNet (Niet et al., 2021)
        x = pd.read_csv("dataset/ihdp/ihdp.csv", usecols=range(2, 27))
        x = torch.FloatTensor(x.to_numpy())

        # normalize the data
        # !mauricio: really weird normalization
        for _ in range(x.shape[1]):
            minval = (x[:, _]).min()
            maxval = (x[:, _]).max()
            x[:, _] = (x[:, _] - minval) / maxval

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
    elif dataset == "news-B":  # IHDP modification in SCIGAN (Bica et al., 2020)
        raise NotImplementedError
    elif dataset == "tcga-B":  # TCGA modification in SCIGAN (Bica et al., 2020)
        raise NotImplementedError
    elif dataset == "sim-T":  # Simulated data in E2B (Taha Bahadori et al., 2022)
        raise NotImplementedError
    else:
        raise ValueError(dataset)


def support(dataset: str) -> tuple[callable[Tensor]]:
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
    elif dataset == "news-B":  # IHDP modification in SCIGAN (Bica et al., 2020)
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
    *args,
    **kwargs,
) -> dict[Tensor]:
    """
    delta_std is the number of standard deviations to reduce from the treatment
    n_train, n_test only useful for simulated datasets
    args, kwargs are passed to load_data
    """
    # -- should be same as vcnet code, but vectorized -- #

    D = load_data(dataset, *args, **kwargs)
    x, t, train_ix, test_ix = D["x"], D["t"], D["train_ix"], D["test_ix"]
    y, noise = outcome(t, x, dataset)

    train_matrix = cat([t[train_ix, None], x[train_ix], y[train_ix, None]], dim=1)
    test_matrix = cat([t[test_ix, None], x[test_ix], y[test_ix, None]], dim=1)

    # -- specific to stochastic interventions -- #
    supp = support(dataset)

    if supp == "unit":
        l = t.logit()  # logits only when treatment in (0, 1)
        delta_scale = l.std()
        shifted_treatments = [(l - delta_scale * d).sigmoid() for d in delta_list]
    elif supp == "real":
        delta_scale = t.std()
        shifted_treatments = [(l - delta_scale * d) for d in delta_list]
    else:
        raise NotImplementedError

    # make counterfactuals and shift-response curves
    cfs = stack([outcome(t, x, noise) for t in shifted_treatments], dim=1)
    srf_train = cfs[train_ix, :].mean(0)
    srf_test = cfs[test_ix, :].mean(0)

    return {
        "train_matrix": train_matrix,
        "test_matrix": test_matrix,
        "srf_train": srf_train,
        "srf_test": srf_test,
        "delta_scale": delta_scale,
    }


def Max(*args):
    """point wise max of tensors"""
    return stack(list(args), dim=1).amax(dim=1)


def Min(*args):
    """point wise min of tensors"""
    return stack(list(args), dim=1).amin(dim=1)
