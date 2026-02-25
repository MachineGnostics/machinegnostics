"""Public API for the Magnet neural-network library.

Author: Nirmal Parmar

Notes:
- ANN-focused implementation with Keras-like `Sequential` training flow.
- Organized for future growth into CNN/RNN/Transformer architectures.
"""


# Core modules
from . import activation, data, layers, loss, models, optim, w_init
from .callbacks.early_stopping import EarlyStopping
from .callbacks.lr_scheduler import LRScheduler
from .model_io import save_weights, load_weights
from .device import Device, get_default_device, is_gpu_available
from .models import Sequential
from .random import set_seed
from .tensor import Tensor

# Expose all core layers, activations, and losses at top-level for user convenience
from .layers import Dense, Dropout, BatchNorm1d
from .activation import ReLU, Sigmoid, Tanh
from .loss import MSELoss, BCELoss, CrossEntropyLoss, MAELoss
from .optim import SGD, Adam

__all__ = [
	"Tensor",
	"Sequential",
	"Device",
	"get_default_device",
	"is_gpu_available",
	"set_seed",
	"Dense", "Dropout", "BatchNorm1d",
	"ReLU", "Sigmoid", "Tanh",
	"MSELoss", "BCELoss", "CrossEntropyLoss", "MAELoss",
	"SGD", "Adam",
	"layers", "activation", "loss", "optim", "w_init", "data", "models",
	"EarlyStopping", "LRScheduler",
	"save_weights", "load_weights",
]