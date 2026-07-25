"""One initializer."""

from __future__ import annotations

import numpy as np

from .base import Initializer


class Ones(Initializer):
    def __call__(self, shape: tuple[int, ...]) -> np.ndarray:
        return np.ones(shape, dtype=np.float64)
