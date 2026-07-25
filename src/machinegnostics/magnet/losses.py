"""Loss functions for ANN training."""

from __future__ import annotations

from typing import Callable, Union

import numpy as np

from .tensor import Tensor


def mse(y_true: Tensor, y_pred: Tensor) -> Tensor:
    return ((y_true - y_pred) ** 2).mean()


def mae(y_true: Tensor, y_pred: Tensor) -> Tensor:
    return (y_true - y_pred).abs().mean()


def binary_cross_entropy(y_true: Tensor, y_pred: Tensor, eps: float = 1e-7) -> Tensor:
    safe_pred = y_pred + eps
    safe_one_minus = (1.0 - y_pred) + eps
    return -(y_true * safe_pred.log() + (1.0 - y_true) * safe_one_minus.log()).mean()


def categorical_cross_entropy(y_true: Tensor, y_pred: Tensor, eps: float = 1e-7) -> Tensor:
    safe_pred = y_pred + eps
    return -(y_true * safe_pred.log()).sum(axis=-1).mean()


LossLike = Union[str, Callable[[Tensor, Tensor], Tensor]]

_LOSSES = {
    "mse": mse,
    "mean_squared_error": mse,
    "mae": mae,
    "mean_absolute_error": mae,
    "binary_crossentropy": binary_cross_entropy,
    "categorical_crossentropy": categorical_cross_entropy,
}


def get_loss(loss: LossLike) -> Callable[[Tensor, Tensor], Tensor]:
    """Resolve a loss specification into a callable."""

    if callable(loss):
        return loss
    try:
        return _LOSSES[loss.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown loss: {loss}") from exc
