"""Glorot/Xavier uniform initializer."""

from __future__ import annotations

import math

import numpy as np

from .base import Initializer


class GlorotUniform(Initializer):
    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def __call__(self, shape: tuple[int, ...]) -> np.ndarray:
        if len(shape) < 2:
            limit = 1.0
        else:
            fan_in, fan_out = shape[0], shape[1]
            limit = math.sqrt(6.0 / (fan_in + fan_out))
        return self.rng.uniform(-limit, limit, size=shape).astype(np.float64)
