"""Initializer base class and registry."""

from __future__ import annotations

from typing import Callable, Union

import numpy as np

InitializerLike = Union[str, Callable[[tuple[int, ...]], np.ndarray], "Initializer"]


class Initializer:
    """Callable object that produces NumPy arrays for parameter tensors."""

    def __call__(self, shape: tuple[int, ...]) -> np.ndarray:
        raise NotImplementedError


def get_initializer(initializer: InitializerLike | None) -> Callable[[tuple[int, ...]], np.ndarray]:
    """Resolve initializer specifications into callables."""

    from .constant import Constant
    from .glorot_uniform import GlorotUniform
    from .he_normal import HeNormal
    from .ones import Ones
    from .random_normal import RandomNormal
    from .zeros import Zeros

    if initializer is None:
        return GlorotUniform()
    if isinstance(initializer, Initializer):
        return initializer
    if callable(initializer):
        return initializer

    name = initializer.lower()
    registry = {
        "zeros": Zeros(),
        "ones": Ones(),
        "constant": Constant(0.0),
        "random_normal": RandomNormal(),
        "glorot_uniform": GlorotUniform(),
        "he_normal": HeNormal(),
    }
    try:
        return registry[name]
    except KeyError as exc:
        raise ValueError(f"Unknown initializer: {initializer}") from exc
