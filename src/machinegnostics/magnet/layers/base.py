"""Base layer class."""

from __future__ import annotations

from typing import List

import logging

from machinegnostics.magcal.util.logging import get_logger

from ..tensor import Tensor


class Layer:
    """Base class for user-facing layers."""

    def __init__(self, name: str | None = None, trainable: bool = True, verbose: bool = False):
        self.name = name or self.__class__.__name__.lower()
        self.trainable = trainable
        self.built = False
        self.verbose = verbose
        self.logger = get_logger(self.__class__.__name__, logging.INFO if self.verbose else logging.WARNING)
        self.logger.info(f"{self.__class__.__name__} initialized.")

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
