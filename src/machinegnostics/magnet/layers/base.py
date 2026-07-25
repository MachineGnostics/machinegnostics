"""Base layer class."""

from __future__ import annotations

from typing import List

from ..tensor import Tensor


class Layer:
    """Base class for user-facing layers."""

    def __init__(self, name: str | None = None, trainable: bool = True):
        self.name = name or self.__class__.__name__.lower()
        self.trainable = trainable
        self.built = False

    @property
    def params(self) -> List[Tensor]:
        return []

    def build(self, input_shape):
        self.built = True

    def forward(self, inputs: Tensor, training: bool = True) -> Tensor:
        raise NotImplementedError

    def __call__(self, inputs: Tensor, training: bool = True) -> Tensor:
        if not self.built:
            self.build(inputs.shape)
        return self.forward(inputs, training=training)
