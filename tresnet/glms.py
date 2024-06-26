from abc import ABC, abstractmethod

import random
from typing import Callable
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, Generator


class GLMFamily(ABC):
    """Simple class specifying an error model, link function, and inverse link"""

    @abstractmethod
    def link(self, x: Tensor) -> Tensor:
        """Link function that described the relation between the mean and the linear predictor eta.
        Specifically, E[Y] = link(eta)"""
        pass

    @abstractmethod
    def inverse_link(self, x: Tensor) -> Tensor:
        """Inverse of the link function such that eta = inverse_link(E[Y]) where eta is the linear predictor"""
        pass

    @abstractmethod
    def loss(
        self, linear_predictor: Tensor, target: Tensor, reduction: str = "none"
    ) -> Tensor:
        """Loss function that describes the error model"""
        pass

    @abstractmethod
    def sampler(self, generator: Generator, **kwargs) -> Callable[[Tensor], Tensor]:
        """Function that samples from the error model. Must follow the same logidc
        as torch sampling functions. See torch.normal for an example."""
        pass

    def sample_from_linear_predictor(
        self,
        linear_predictor: Tensor,
        seed: int | None = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor] | Tensor:
        """Sample from the error model.

        :param linear_predictor: Linear predictor
        :param randomness: Noise to add to the linear predictor. If None, the the noise is returned along with the sample.
            The main use of the noise is to ensure the same noise is used for multiple samples
        :return: Return a tuple (sample, seeed) where seed is the seed used to generate the sample.
            Using the same seed will yield consisten results for counterfactual curves.
        """
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        gen = torch.Generator().manual_seed(seed)
        out = self.sampler(gen, **kwargs)(linear_predictor)

        if seed is None:
            return out, seed
        else:
            return out


class Gaussian(GLMFamily):
    """Gaussian error model"""

    def link(self, x: Tensor) -> Tensor:
        return x

    def inverse_link(self, x: Tensor) -> Tensor:
        return x

    def loss(
        self, linear_predictor: Tensor, target: Tensor, reduction: str = "none"
    ) -> Tensor:
        if linear_predictor.shape[1] == 1 and target.shape[1] > 1:
            linear_predictor = linear_predictor.repeat(1, target.shape[1])
        elif target.shape[1] == 1 and linear_predictor.shape[1] > 1:
            target = target.repeat(1, linear_predictor.shape[1])
        return F.mse_loss(linear_predictor, target, reduction=reduction)

    def sampler(self, generator: torch.Generator, noise_scale: float = 1.0) -> callable:
        return lambda lp: torch.normal(lp, noise_scale, generator=generator)


class Bernoulli(GLMFamily):
    """Bernoulli error model"""

    def link(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x)

    def inverse_link(self, x: Tensor) -> Tensor:
        return torch.logit(x.clamp(1e-6, 1 - 1e-6))

    def loss(
        self, linear_predictor: Tensor, target: Tensor, reduction: str = "none"
    ) -> Tensor:
        if linear_predictor.shape[1] == 1 and target.shape[1] > 1:
            linear_predictor = linear_predictor.repeat(1, target.shape[1])
        elif target.shape[1] == 1 and linear_predictor.shape[1] > 1:
            target = target.repeat(1, linear_predictor.shape[1])
        return F.binary_cross_entropy_with_logits(
            linear_predictor, target, reduction=reduction
        )

    def sampler(self, generator: torch.Generator) -> callable:
        return lambda lp: torch.bernoulli(torch.sigmoid(lp))


class Poisson(GLMFamily):
    """Poisson error model"""

    def __init__(self, offset: float = 0.0):
        super().__init__()
        self.off = offset

    def link(self, x: Tensor) -> Tensor:
        return torch.exp((x + self.off).clamp(min=-7, max=10))

    def inverse_link(self, x: Tensor) -> Tensor:
        # y = torch.log(x + 1e-6)
        # # revert symlog from link
        # return torch.sign(y) * (torch.exp(torch.abs(y)) - 1) / 0.1
        return torch.log(x + 1e-3) - self.off

    def loss(
        self,
        linear_predictor: Tensor,
        target: Tensor,
        reduction: str = "none",
        min=-7,
        max=7,
    ) -> Tensor:
        if linear_predictor.shape[1] == 1 and target.shape[1] > 1:
            linear_predictor = linear_predictor.repeat(1, target.shape[1])
        elif target.shape[1] == 1 and linear_predictor.shape[1] > 1:
            target = target.repeat(1, linear_predictor.shape[1])
        lp = (linear_predictor + self.off).clamp(min=-7, max=10)
        return F.poisson_nll_loss(
            lp, target, log_input=True, reduction=reduction, full=True
        )

    def sampler(self, generator: torch.Generator) -> callable:
        return lambda l: torch.poisson(torch.exp(l), generator=generator)
