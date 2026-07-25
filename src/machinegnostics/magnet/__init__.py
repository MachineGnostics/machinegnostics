"""Machine Gnostics magnet package.

The package exposes ANN building blocks directly from `machinegnostics.magnet`.
"""

from .gn_activations import ActivationFunctions
from .neuron import GnosticNeuron
from .tensor import Tensor
from .model import Model, Sequential
from .history import History
from .losses import (
	get_loss,
	gnostic_weighted_mse,
	gnostic_weighted_rmse,
	fidelity_loss,
	infidelity_loss,
	irrelevance_loss,
	relevance_loss,
	gnostic_characteristic_loss,
)
from .activations import get_activation, fi, fj, hi, hj
from .optimizers import get_optimizer, SGD, Adam
from .initializers import get_initializer
from .callbacks import Callback, EarlyStopping
from .layers import Layer, Dense, iDense, jDense

__all__ = [
	"ActivationFunctions",
	"GnosticNeuron",
	"Tensor",
	"Model",
	"Sequential",
	"History",
	"get_loss",
	"get_activation",
	"fi",
	"fj",
	"hi",
	"hj",
	"gnostic_weighted_mse",
	"gnostic_weighted_rmse",
	"fidelity_loss",
	"infidelity_loss",
	"irrelevance_loss",
	"relevance_loss",
	"gnostic_characteristic_loss",
	"get_optimizer",
	"SGD",
	"Adam",
	"get_initializer",
	"Callback",
	"EarlyStopping",
	"Layer",
	"Dense",
	"iDense",
	"jDense",
]