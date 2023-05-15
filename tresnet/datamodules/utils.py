import torch


def Max(*args):
    """point wise max of tensors"""
    return torch.stack(list(args), dim=1).amax(dim=1)


def Min(*args):
    """point wise min of tensors"""
    return torch.stack(list(args), dim=1).amin(dim=1)
