"""Glorot uniform (Xavier uniform) weight initialization."""

from __future__ import annotations

import numpy as np

from .base import Initializer, compute_fans, normalize_shape


class GlorotUniform(Initializer):
    """Sample weights from the Glorot uniform distribution.

    Glorot uniform, also known as Xavier uniform, draws values from
    ``U(-limit, limit)`` where ``limit = sqrt(6 / (fan_in + fan_out))``.
    This balances variance between the forward and backward pass and is a
    strong default for dense layers with tanh- or sigmoid-like activations.
    """

    def __init__(self, seed=None, dtype=np.float64):
        """Create a Glorot uniform initializer."""
        super().__init__(dtype=dtype)
        self.rng = np.random.default_rng(seed)

    def __call__(self, shape) -> np.ndarray:
        """Return an array initialized with Glorot uniform scaling."""
        normalized = normalize_shape(shape, min_ndim=2)
        fan_in, fan_out = compute_fans(normalized)
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return self.rng.uniform(-limit, limit, size=normalized).astype(self.dtype, copy=False)
