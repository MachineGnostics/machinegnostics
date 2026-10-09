"""Base classes and helpers for weight initializers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np


def normalize_shape(shape: int | Iterable[int], *, min_ndim: int = 1) -> tuple[int, ...]:
    """Return ``shape`` as a validated tuple of positive integers.

    Parameters
    ----------
    shape : int or iterable of int
        Requested tensor shape.
    min_ndim : int, default=1
        Minimum number of dimensions required by the caller.

    Returns
    -------
    tuple of int
        Validated shape tuple.

    Raises
    ------
    TypeError
        If ``shape`` cannot be interpreted as integers.
    ValueError
        If the shape is empty, too small, or contains non-positive values.
    """
    if isinstance(shape, int):
        shape = (shape,)
    elif not isinstance(shape, tuple):
        shape = tuple(shape)

    if not shape:
        raise ValueError("Shape must contain at least one dimension")
    if len(shape) < min_ndim:
        raise ValueError(f"Expected at least {min_ndim} dimensions, got shape {shape}")

    normalized: list[int] = []
    for dimension in shape:
        value = int(dimension)
        if value <= 0:
            raise ValueError(f"Shape dimensions must be positive, got {shape}")
        normalized.append(value)
    return tuple(normalized)


def compute_fans(shape: int | Iterable[int]) -> tuple[int, int]:
    """Compute fan-in and fan-out for a weight tensor shape.

    Parameters
    ----------
    shape : int or iterable of int
        Shape of the target weight tensor. Two-dimensional shapes are treated
        as ``(fan_in, fan_out)``. Higher-rank shapes follow the common kernel
        convention where the last two dimensions represent input and output
        channels and preceding dimensions form the receptive field.

    Returns
    -------
    tuple of int
        ``(fan_in, fan_out)`` for the provided shape.
    """
    normalized = normalize_shape(shape, min_ndim=2)
    if len(normalized) == 2:
        return normalized[0], normalized[1]

    receptive_field = int(np.prod(normalized[:-2]))
    fan_in = normalized[-2] * receptive_field
    fan_out = normalized[-1] * receptive_field
    return fan_in, fan_out


class Initializer(ABC):
    """Base class for MAGNET weight initializers.

    Weight initialization strongly influences optimization stability,
    convergence speed, and how well gradients propagate through deep networks.
    Concrete subclasses implement strategies suited to different activations
    and architectures.

    Examples
    --------
    Common choices include:

    - :class:`GlorotUniform` for tanh/sigmoid style networks
    - :class:`HeNormal` or :class:`HeUniform` for ReLU-family activations
    - :class:`Zeros` for bias vectors
    """

    def __init__(self, dtype=np.float64):
        """Create an initializer with a target NumPy dtype."""
        self.dtype = np.dtype(dtype)

    @abstractmethod
    def __call__(self, shape: int | Iterable[int]) -> np.ndarray:
        """Initialize an array with the provided shape.

        Parameters
        ----------
        shape : int or iterable of int
            Shape of the requested parameter tensor.

        Returns
        -------
        np.ndarray
            Initialized weight array.
        """

