"""Machine Gnostics magnet package.

The package now exposes ANN building blocks directly from `machinegnostics.magnet`
so users do not need to import from `machinegnostics.magnet.nn`.
"""

from .gn_activations import ActivationFunctions
from .neuron import GnosticNeuron
from .tensor import Tensor
from .model import Model, Sequential
from .history import History
from .losses import get_loss
from .activations import get_activation
from .optimizers import get_optimizer, SGD, Adam
from .initializers import get_initializer
from .callbacks import Callback, EarlyStopping
from .layers import Layer, Dense

__all__ = [
	"ActivationFunctions",
	"GnosticNeuron",
	"Tensor",
	"Model",
	"Sequential",
	"History",
	"get_loss",
	"get_activation",
	"get_optimizer",
	"SGD",
	"Adam",
	"get_initializer",
	"Callback",
	"EarlyStopping",
	"Layer",
	"Dense",
]