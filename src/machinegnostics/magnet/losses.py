"""Loss functions for ANN training."""

from __future__ import annotations

from typing import Callable, Union

import numpy as np

from .tensor import Tensor, _compute_gnostic_bundle


def _as_tensor(value) -> Tensor:
    return value if isinstance(value, Tensor) else Tensor(value, requires_grad=False)


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


def gnostic_weighted_mse(y_true: Tensor, y_pred: Tensor, scale_param: str | float = "auto") -> Tensor:
    error = _as_tensor(y_pred) - _as_tensor(y_true)
    bundle = _compute_gnostic_bundle(error.data, scale_param=scale_param)
    weights = Tensor(bundle["weights"], requires_grad=False)
    return ((error * weights) ** 2).mean()


def gnostic_weighted_rmse(y_true: Tensor, y_pred: Tensor, scale_param: str | float = "auto") -> Tensor:
    return gnostic_weighted_mse(y_true, y_pred, scale_param=scale_param) ** 0.5


def fidelity_loss(y_true: Tensor, y_pred: Tensor, scale_param: str | float = "auto") -> Tensor:
    bundle = _compute_gnostic_bundle((_as_tensor(y_pred) - _as_tensor(y_true)).data, scale_param=scale_param)
    return Tensor(float(np.mean(bundle["fi"])), requires_grad=False)


def infidelity_loss(y_true: Tensor, y_pred: Tensor, scale_param: str | float = "auto") -> Tensor:
    bundle = _compute_gnostic_bundle((_as_tensor(y_pred) - _as_tensor(y_true)).data, scale_param=scale_param)
    return Tensor(float(np.mean(bundle["fj"])), requires_grad=False)


def irrelevance_loss(y_true: Tensor, y_pred: Tensor, scale_param: str | float = "auto") -> Tensor:
    bundle = _compute_gnostic_bundle((_as_tensor(y_pred) - _as_tensor(y_true)).data, scale_param=scale_param)
    return Tensor(float(np.mean(bundle["hi"] ** 2)), requires_grad=False)


def relevance_loss(y_true: Tensor, y_pred: Tensor, scale_param: str | float = "auto") -> Tensor:
    bundle = _compute_gnostic_bundle((_as_tensor(y_pred) - _as_tensor(y_true)).data, scale_param=scale_param)
    return Tensor(float(np.mean(bundle["hj"] ** 2)), requires_grad=False)


def gnostic_characteristic_loss(y_true: Tensor, y_pred: Tensor, scale_param: str | float = "auto") -> Tensor:
    bundle = _compute_gnostic_bundle((_as_tensor(y_pred) - _as_tensor(y_true)).data, scale_param=scale_param)
    scalar = float(np.mean([np.mean(bundle["fi"]), np.mean(bundle["fj"]), np.mean(bundle["hi"]), np.mean(bundle["hj"])]))
    return Tensor(scalar, requires_grad=False)


LossLike = Union[str, Callable[[Tensor, Tensor], Tensor]]

_LOSSES = {
    "mse": mse,
    "mean_squared_error": mse,
    "mae": mae,
    "mean_absolute_error": mae,
    "binary_crossentropy": binary_cross_entropy,
    "categorical_crossentropy": categorical_cross_entropy,
    "gnostic_weighted_mse": gnostic_weighted_mse,
    "gwmse": gnostic_weighted_mse,
    "gnostic_weighted_rmse": gnostic_weighted_rmse,
    "gwrmse": gnostic_weighted_rmse,
    "fidelity": fidelity_loss,
    "fi": fidelity_loss,
    "infidelity": infidelity_loss,
    "fj": infidelity_loss,
    "irrelevance": irrelevance_loss,
    "hi": irrelevance_loss,
    "relevance": relevance_loss,
    "hj": relevance_loss,
    "gnostic_characteristic_loss": gnostic_characteristic_loss,
    "gnostic_loss": gnostic_characteristic_loss,
}


def get_loss(loss: LossLike) -> Callable[[Tensor, Tensor], Tensor]:
    """Resolve a loss specification into a callable."""

    if callable(loss):
        return loss
    try:
        return _LOSSES[loss.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown loss: {loss}") from exc
