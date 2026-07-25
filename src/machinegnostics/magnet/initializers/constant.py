"""Constant initializer."""

from __future__ import annotations

import numpy as np

from .base import Initializer


class Constant(Initializer):
    def __init__(self, value: float = 0.0):
        self.value = value

    def __call__(self, shape: tuple[int, ...]) -> np.ndarray:
        return np.full(shape, self.value, dtype=np.float64)
