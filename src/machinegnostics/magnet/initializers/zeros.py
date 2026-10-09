"""Zero initializer for MAGNET."""

from __future__ import annotations

import numpy as np

from .base import Initializer, normalize_shape


class Zeros(Initializer):
    """Return arrays filled with zeros."""

    def __call__(self, shape) -> np.ndarray:
        """Return a zero-filled array."""
        normalized = normalize_shape(shape)
        return np.zeros(normalized, dtype=self.dtype)
