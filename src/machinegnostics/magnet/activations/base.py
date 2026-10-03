"""Base activation layer for MAGNET."""

from __future__ import annotations

from ..layers.base import Layer


class Activation(Layer):
    """Base class for MAGNET activation layers."""

    def __init__(self, name: str | None = None, verbose: bool = False):
        super().__init__(name=name, verbose=verbose)

    def forward(self, x, training: bool = True):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError(
            "Activation layers use torch autograd. Call backward() on the output tensor instead."
        )
