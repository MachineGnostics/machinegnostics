"""Miscellaneous helper functions for MAGNET."""

from __future__ import annotations

import numpy as np


def set_random_seed(seed: int | None) -> None:
    """Seed NumPy's global random generator for reproducible experiments."""
    if seed is not None:
        np.random.seed(int(seed))


def ensure_2d(array) -> np.ndarray:
    """Return an array with at least two dimensions.

    One-dimensional inputs become column vectors with shape ``(n, 1)``.
    """
    value = np.asarray(array)
    if value.ndim == 0:
        return value.reshape(1, 1)
    if value.ndim == 1:
        return value.reshape(-1, 1)
    return value


def validate_same_length(*arrays) -> None:
    """Validate that all provided arrays share the same leading dimension."""
    lengths = [len(np.asarray(array)) for array in arrays]
    if len(set(lengths)) > 1:
        raise ValueError(f"Expected arrays with matching lengths, got {lengths}")


__all__ = ["set_random_seed", "ensure_2d", "validate_same_length"]
