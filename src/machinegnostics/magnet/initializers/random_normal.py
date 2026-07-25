"""Random normal initializer."""

from __future__ import annotations

import numpy as np

from .base import Initializer


class RandomNormal(Initializer):
    def __init__(self, mean: float = 0.0, stddev: float = 0.05, seed: int | None = None):
        self.mean = mean
        self.stddev = stddev
        self.rng = np.random.default_rng(seed)

    def __call__(self, shape: tuple[int, ...]) -> np.ndarray:
        return self.rng.normal(self.mean, self.stddev, size=shape).astype(np.float64)
