"""Configurable normal initializer for MAGNET."""

from __future__ import annotations

import numpy as np

from .base import Initializer, normalize_shape


class Normal(Initializer):
    """Sample weights from a configurable Gaussian distribution."""

    def __init__(self, mean: float = 0.0, stddev: float = 0.01, seed=None, dtype=np.float64):
        """Initialize the configurable normal sampler."""
        super().__init__(dtype=dtype)
        self.mean = float(mean)
        self.stddev = float(stddev)
        self.rng = np.random.default_rng(seed)

    def __call__(self, shape) -> np.ndarray:
        """Return samples from the configured normal distribution."""
        normalized = normalize_shape(shape)
        return self.rng.normal(self.mean, self.stddev, size=normalized).astype(self.dtype, copy=False)
