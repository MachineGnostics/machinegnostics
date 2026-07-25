"""Zero initializer."""

from __future__ import annotations

import numpy as np

from .base import Initializer


class Zeros(Initializer):
    def __call__(self, shape: tuple[int, ...]) -> np.ndarray:
        return np.zeros(shape, dtype=np.float64)
