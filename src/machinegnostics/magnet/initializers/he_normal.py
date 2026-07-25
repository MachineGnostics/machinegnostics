"""He normal initializer."""

from __future__ import annotations

import math

import numpy as np

from .base import Initializer


class HeNormal(Initializer):
    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def __call__(self, shape: tuple[int, ...]) -> np.ndarray:
        if len(shape) < 2:
            stddev = 1.0
        else:
            fan_in = shape[0]
            stddev = math.sqrt(2.0 / fan_in)
        return self.rng.normal(0.0, stddev, size=shape).astype(np.float64)
