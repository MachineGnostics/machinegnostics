"""Weight initializers for Magnet layers.

Author: Nirmal Parmar

Notes:
- Initializers accept `shape` and optional `fan_in`, `fan_out`.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np


def zeros(shape, **kwargs):
    return np.zeros(shape, dtype=np.float64)


def ones(shape, **kwargs):
    return np.ones(shape, dtype=np.float64)


def normal(shape, mean=0.0, std=0.02, **kwargs):
    return np.random.normal(loc=mean, scale=std, size=shape).astype(np.float64)


def uniform(shape, low=-0.05, high=0.05, **kwargs):
    return np.random.uniform(low=low, high=high, size=shape).astype(np.float64)


def xavier_uniform(shape, fan_in=None, fan_out=None, **kwargs):
    if fan_in is None or fan_out is None:
        if len(shape) < 2:
            raise ValueError("xavier_uniform requires fan_in and fan_out for 1D shapes.")
        fan_in, fan_out = shape[-2], shape[-1]
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=shape).astype(np.float64)


def xavier_normal(shape, fan_in=None, fan_out=None, **kwargs):
    if fan_in is None or fan_out is None:
        if len(shape) < 2:
            raise ValueError("xavier_normal requires fan_in and fan_out for 1D shapes.")
        fan_in, fan_out = shape[-2], shape[-1]
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.normal(0.0, std, size=shape).astype(np.float64)


def kaiming_uniform(shape, fan_in=None, nonlinearity="relu", **kwargs):
    if fan_in is None:
        if len(shape) < 2:
            raise ValueError("kaiming_uniform requires fan_in for 1D shapes.")
        fan_in = shape[-2]
    gain = np.sqrt(2.0) if nonlinearity in {"relu", "leaky_relu"} else 1.0
    bound = np.sqrt(3.0) * gain / np.sqrt(fan_in)
    return np.random.uniform(-bound, bound, size=shape).astype(np.float64)


def kaiming_normal(shape, fan_in=None, nonlinearity="relu", **kwargs):
    if fan_in is None:
        if len(shape) < 2:
            raise ValueError("kaiming_normal requires fan_in for 1D shapes.")
        fan_in = shape[-2]
    gain = np.sqrt(2.0) if nonlinearity in {"relu", "leaky_relu"} else 1.0
    std = gain / np.sqrt(fan_in)
    return np.random.normal(0.0, std, size=shape).astype(np.float64)


_INITIALIZERS: Dict[str, Callable] = {
    "zeros": zeros,
    "ones": ones,
    "normal": normal,
    "uniform": uniform,
    "xavier_uniform": xavier_uniform,
    "xavier_normal": xavier_normal,
    "kaiming_uniform": kaiming_uniform,
    "kaiming_normal": kaiming_normal,
}


def get_initializer(name: str) -> Callable:
    """Resolve initializer name to callable."""
    key = name.lower().strip()
    if key not in _INITIALIZERS:
        available = ", ".join(sorted(_INITIALIZERS.keys()))
        raise ValueError(f"Unknown initializer '{name}'. Available: {available}")
    return _INITIALIZERS[key]


__all__ = [
    "zeros",
    "ones",
    "normal",
    "uniform",
    "xavier_uniform",
    "xavier_normal",
    "kaiming_uniform",
    "kaiming_normal",
    "get_initializer",
]