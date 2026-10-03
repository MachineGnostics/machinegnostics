"""He uniform weight initialization."""

from __future__ import annotations

import numpy as np

from .base import Initializer, compute_fans, normalize_shape


class HeUniform(Initializer):
    """Sample weights from the He uniform distribution.

    He uniform draws values from ``U(-limit, limit)`` with
    ``limit = sqrt(6 / fan_in)``. It preserves activation scale better for
    ReLU-family networks than Glorot-style schemes because it focuses on the
    number of input connections only.
    """

    def __init__(self, seed=None, dtype=np.float64):
        """Create a He uniform initializer."""
        super().__init__(dtype=dtype)
        self.rng = np.random.default_rng(seed)

    def __call__(self, shape) -> np.ndarray:
        """Return an array initialized with He uniform scaling."""
        normalized = normalize_shape(shape, min_ndim=2)
        fan_in, _ = compute_fans(normalized)
        limit = np.sqrt(6.0 / fan_in)
        return self.rng.uniform(-limit, limit, size=normalized).astype(self.dtype, copy=False)
