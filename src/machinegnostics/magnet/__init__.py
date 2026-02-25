"""Public API for the Magnet neural-network library.

Author: Nirmal Parmar

Notes:
- ANN-focused implementation with Keras-like `Sequential` training flow.
- Organized for future growth into CNN/RNN/Transformer architectures.
"""

from . import acti, data, layers, loss, models, optim, w_init
from .callbacks.early_stopping import EarlyStopping
from .callbacks.lr_scheduler import LRScheduler
from .model_io import save_weights, load_weights
from .device import Device, get_default_device, is_gpu_available
from .models import Sequential
from .random import set_seed
from .tensor import Tensor

__all__ = [
	"Tensor",
	"Sequential",
	"Device",
	"get_default_device",
	"is_gpu_available",
	"set_seed",
	"layers",
	"acti",
	"loss",
	"optim",
	"w_init",
	"data",
	"models",
    "EarlyStopping",
    "LRScheduler",
    "save_weights",
    "load_weights",
]