"""All-zero weight initializer."""

from __future__ import annotations

import numpy as np

from .base import Initializer, normalize_shape


class Zeros(Initializer):
    """Return arrays filled with zeros.

    Zero initialization is commonly used for bias vectors because it starts
    offsets from a neutral value. It is generally not suitable for full weight
    matrices because identical parameters receive identical gradients.
    """

    def __call__(self, shape) -> np.ndarray:
        """Return a zero-filled array with the requested shape."""
        normalized = normalize_shape(shape)
        return np.zeros(normalized, dtype=self.dtype)
