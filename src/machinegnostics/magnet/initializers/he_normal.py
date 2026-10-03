"""He normal weight initialization."""

from __future__ import annotations

import numpy as np

from .base import Initializer, compute_fans, normalize_shape


class HeNormal(Initializer):
    """Sample weights from the He normal distribution.

    He normal draws values from ``N(0, sqrt(2 / fan_in))`` and is especially
    effective for layers followed by rectified activations such as ReLU and
    LeakyReLU. The fan-in-only scaling keeps forward activations and gradients
    from shrinking too aggressively in deep rectifier stacks.
    """

    def __init__(self, seed=None, dtype=np.float64):
        """Create a He normal initializer."""
        super().__init__(dtype=dtype)
        self.rng = np.random.default_rng(seed)

    def __call__(self, shape) -> np.ndarray:
        """Return an array initialized with He normal scaling."""
        normalized = normalize_shape(shape, min_ndim=2)
        fan_in, _ = compute_fans(normalized)
        stddev = np.sqrt(2.0 / fan_in)
        return self.rng.normal(0.0, stddev, size=normalized).astype(self.dtype, copy=False)
