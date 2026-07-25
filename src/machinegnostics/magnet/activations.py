"""Activation functions for ANN layers."""

from __future__ import annotations

from typing import Callable, Dict, Union

import numpy as np

from .tensor import Tensor, _compute_gnostic_bundle

ActivationLike = Union[str, Callable[[Tensor], Tensor], None]


def linear(x: Tensor) -> Tensor:
    return x


def relu(x: Tensor) -> Tensor:
    return x.relu()


def sigmoid(x: Tensor) -> Tensor:
    return x.sigmoid()


def tanh(x: Tensor) -> Tensor:
    return x.tanh()


def softmax(x: Tensor) -> Tensor:
    shifted = x - x.max(axis=-1, keepdims=True)
    exp_values = shifted.exp()
    return exp_values / exp_values.sum(axis=-1, keepdims=True)


def _gnostic_activation(x, activation_type: str, scale_param: str | float = "auto", reference: float | None = None):
    bundle = _compute_gnostic_bundle(x, scale_param=scale_param, activation_type=activation_type, reference=reference)
    if isinstance(x, Tensor):
        return Tensor(bundle["activation"], requires_grad=x.requires_grad)
    return bundle["activation"]


def fi(x, scale_param: str | float = "auto", reference: float | None = None):
    return _gnostic_activation(x, "fi", scale_param=scale_param, reference=reference)


def fj(x, scale_param: str | float = "auto", reference: float | None = None):
    return _gnostic_activation(x, "fj", scale_param=scale_param, reference=reference)


def hi(x, scale_param: str | float = "auto", reference: float | None = None):
    return _gnostic_activation(x, "hi", scale_param=scale_param, reference=reference)


def hj(x, scale_param: str | float = "auto", reference: float | None = None):
    return _gnostic_activation(x, "hj", scale_param=scale_param, reference=reference)


_ACTIVATIONS: Dict[str, Callable[[Tensor], Tensor]] = {
    "linear": linear,
    "relu": relu,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "softmax": softmax,
    "fi": fi,
    "fj": fj,
    "hi": hi,
    "hj": hj,
}


def get_activation(activation: ActivationLike) -> Callable[[Tensor], Tensor]:
    """Resolve an activation specification into a callable."""

    if activation is None:
        return linear
    if callable(activation):
        return activation
    try:
        return _ACTIVATIONS[activation.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown activation: {activation}") from exc
