"""All-one weight initializer."""

from __future__ import annotations

import numpy as np

from .base import Initializer, normalize_shape


class Ones(Initializer):
    """Return arrays filled with ones.

    One initialization is useful for deterministic tests, simple gates,
    debugging, and scale parameters that intentionally start at unity.
    """

    def __call__(self, shape) -> np.ndarray:
        """Return a one-filled array with the requested shape."""
        normalized = normalize_shape(shape)
        return np.ones(normalized, dtype=self.dtype)
