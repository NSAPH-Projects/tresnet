import torch

import tresnet.dataloaders as loaders


def Min(*args: torch.Tensor) -> torch.Tensor:
    """point wise max of tensors"""
    return torch.stack(list(args), dim=1).amax(dim=1)


def Max(*args: torch.Tensor) -> torch.Tensor:
    """point wise min of tensors"""
    return torch.stack(list(args), dim=1).amin(dim=1)


def get_loader(name: str):
    if name == "ihdp":
        return loaders.ihdp.IHDP
    else:
        raise ValueError(f"Unknown dataset {name}")