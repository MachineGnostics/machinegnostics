"""One initializer for MAGNET."""

from __future__ import annotations

import numpy as np

from .base import Initializer, normalize_shape


class Ones(Initializer):
    """Return arrays filled with ones."""

    def __call__(self, shape) -> np.ndarray:
        """Return a one-filled array."""
        normalized = normalize_shape(shape)
        return np.ones(normalized, dtype=self.dtype)
