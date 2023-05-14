from abc import ABC, abstractmethod
import sys

import torch
from torch import Tensor


class Shift(ABC):
    """Base class for a conditional shift functions"""

    @abstractmethod
    def __call__(
        self, input: Tensor, shift: Tensor, cond: Tensor | None = None
    ) -> Tensor:
        """Evaluate the shift function"""
        raise NotImplementedError

    def has_inverse(self) -> bool:
        return getattr(self, "inverse", None) is not None


class Percent(Shift):
    """Shift function that reduces by a constant percentage of the input"""

    def __call__(self, input: Tensor, shift: Tensor) -> Tensor:
        return input * (1 - shift)

    def inverse(self, input: Tensor, shift: Tensor) -> tuple[Tensor, Tensor]:
        return input / (1 - shift), torch.log(1.0 - shift)


class Subtract(Shift):
    """Shift function that adds a constant to the input"""

    def __call__(self, input: Tensor, shift: Tensor) -> Tensor:
        return input - shift

    def inverse(self, input: Tensor, shift: Tensor) -> tuple[Tensor, Tensor]:
        return input + shift, torch.zeros_like(input - shift)


class Cutoff(Shift):
    """Shift function that cutoffs values that exceed a threshold"""

    def __call__(self, input: Tensor, shift: Tensor) -> Tensor:
        return input.clamp(max=shift)
