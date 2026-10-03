"""Glorot normal initializer for MAGNET."""

from __future__ import annotations

import numpy as np

from .base import Initializer, compute_fans, normalize_shape


class GlorotNormal(Initializer):
    """Sample weights from the Glorot normal distribution."""

    def __init__(self, seed=None, dtype=np.float64):
        """Initialize the Glorot normal sampler."""
        super().__init__(dtype=dtype)
        self.rng = np.random.default_rng(seed)

    def __call__(self, shape) -> np.ndarray:
        """Return Glorot-scaled normal samples."""
        normalized = normalize_shape(shape, min_ndim=2)
        fan_in, fan_out = compute_fans(normalized)
        stddev = np.sqrt(2.0 / (fan_in + fan_out))
        return self.rng.normal(0.0, stddev, size=normalized).astype(self.dtype, copy=False)
