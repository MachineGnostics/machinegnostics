"""He uniform initializer for MAGNET."""

from __future__ import annotations

import numpy as np

from .base import Initializer, compute_fans, normalize_shape


class HeUniform(Initializer):
    """Sample weights from the He uniform distribution."""

    def __init__(self, seed=None, dtype=np.float64):
        """Initialize the He uniform sampler."""
        super().__init__(dtype=dtype)
        self.rng = np.random.default_rng(seed)

    def __call__(self, shape) -> np.ndarray:
        """Return He-scaled uniform samples."""
        normalized = normalize_shape(shape, min_ndim=2)
        fan_in, _ = compute_fans(normalized)
        limit = np.sqrt(6.0 / fan_in)
        return self.rng.uniform(-limit, limit, size=normalized).astype(self.dtype, copy=False)
