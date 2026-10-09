"""Configurable uniform initializer for MAGNET."""

from __future__ import annotations

import numpy as np

from .base import Initializer, normalize_shape


class Uniform(Initializer):
    """Sample weights uniformly from a configurable interval."""

    def __init__(self, low: float = -1.0, high: float = 1.0, seed=None, dtype=np.float64):
        """Initialize the configurable uniform sampler."""
        super().__init__(dtype=dtype)
        self.low = float(low)
        self.high = float(high)
        if self.high < self.low:
            raise ValueError('high must be greater than or equal to low')
        self.rng = np.random.default_rng(seed)

    def __call__(self, shape) -> np.ndarray:
        """Return samples from the configured uniform interval."""
        normalized = normalize_shape(shape)
        return self.rng.uniform(self.low, self.high, size=normalized).astype(self.dtype, copy=False)
