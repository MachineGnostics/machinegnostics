"""He normal initializer for MAGNET."""

from __future__ import annotations

import numpy as np

from .base import Initializer, compute_fans, normalize_shape


class HeNormal(Initializer):
    """Sample weights from the He normal distribution."""

    def __init__(self, seed=None, dtype=np.float64):
        """Initialize the He normal sampler."""
        super().__init__(dtype=dtype)
        self.rng = np.random.default_rng(seed)

    def __call__(self, shape) -> np.ndarray:
        """Return He-scaled normal samples."""
        normalized = normalize_shape(shape, min_ndim=2)
        fan_in, _ = compute_fans(normalized)
        stddev = np.sqrt(2.0 / fan_in)
        return self.rng.normal(0.0, stddev, size=normalized).astype(self.dtype, copy=False)
