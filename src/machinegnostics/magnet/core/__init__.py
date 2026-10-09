"""Core MAGNET runtime building blocks.

Developer note
-------------
Author: Nirmal Parmar, Machine Gnostics

This package contains the hidden runtime helpers that power the public
MAGNET API. It is intentionally small and avoids importing the model package
eagerly so that the backend can stay lightweight and easier to debug.
"""

from .runtime import RuntimeConfig, configure, get_runtime, get_torch_device, get_torch_dtype, to_numpy, to_torch
from .tensor import Tensor, unbroadcast
from .history import History
from .callbacks import Callback, EarlyStopping

__all__ = [
	"RuntimeConfig",
	"configure",
	"get_runtime",
	"get_torch_device",
	"get_torch_dtype",
	"to_numpy",
	"to_torch",
	"Tensor",
	"unbroadcast",
	"History",
	"Callback",
	"EarlyStopping",
]