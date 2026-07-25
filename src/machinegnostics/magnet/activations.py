"""Activation functions for ANN layers."""

from __future__ import annotations

from typing import Callable, Dict, Union

from .tensor import Tensor

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


_ACTIVATIONS: Dict[str, Callable[[Tensor], Tensor]] = {
    "linear": linear,
    "relu": relu,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "softmax": softmax,
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
