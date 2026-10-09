"""Glorot uniform initializer for MAGNET."""

from __future__ import annotations

import numpy as np

from .base import Initializer, compute_fans, normalize_shape


class GlorotUniform(Initializer):
    """Sample weights from the Glorot uniform distribution."""

    def __init__(self, seed=None, dtype=np.float64):
        """Initialize the Glorot uniform sampler."""
        super().__init__(dtype=dtype)
        self.rng = np.random.default_rng(seed)

    def __call__(self, shape) -> np.ndarray:
        """Return Glorot-scaled uniform samples."""
        normalized = normalize_shape(shape, min_ndim=2)
        fan_in, fan_out = compute_fans(normalized)
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return self.rng.uniform(-limit, limit, size=normalized).astype(self.dtype, copy=False)
