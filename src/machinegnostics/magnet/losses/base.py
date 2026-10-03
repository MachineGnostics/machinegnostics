"""Base loss utilities for MAGNET."""

from __future__ import annotations

import logging

from ..core.tensor import Tensor
from ..utils.logging import get_logger


class Loss:
    """Base class for MAGNET losses."""

    def __init__(self, name: str | None = None, verbose: bool = False):
        self.name = name or self.__class__.__name__
        self.verbose = verbose
        self.logger = get_logger(self.name, logging.INFO if verbose else logging.WARNING)

    def forward(self, y_pred, y_true):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError(
            "Loss functions use torch autograd. Call backward() on the returned tensor instead."
        )

    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)


def prepare_tensors(y_pred, y_true) -> tuple[Tensor, Tensor]:
    y_pred = y_pred if isinstance(y_pred, Tensor) else Tensor(y_pred)
    y_true = y_true if isinstance(y_true, Tensor) else Tensor(y_true)
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Shape mismatch: y_pred {y_pred.shape} != y_true {y_true.shape}")
    return y_pred, y_true
