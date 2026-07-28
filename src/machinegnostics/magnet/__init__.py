"""Machine Gnostics magnet package.

Tensor-based neural network primitives for the Machine Gnostics project.
"""

from .tensor import Tensor
from .history import History
from .initializers import get_initializer, Initializer, Zeros, Ones, RandomNormal, XavierUniform, HeNormal
from .activations import (
	get_activation,
	fi,
	fj,
	hi,
	hj,
	Activation,
	ReLU,
	Sigmoid,
	Tanh,
	Softmax,
	Fidelity,
	Infidelity,
	Irrelevance,
	Relevance,
)
from .losses import (
	get_loss,
	Loss,
	MSE,
	BinaryCrossEntropy,
	GnosticFidelity,
	GnosticInfidelity,
	GnosticInformation,
	GnosticResidualEntropy,
	GnosticMSE,
	GnosticBinaryCrossEntropy,
	gnostic_weighted_mse,
	gnostic_weighted_rmse,
	fidelity_loss,
	infidelity_loss,
	irrelevance_loss,
	relevance_loss,
	gnostic_characteristic_loss,
)
from .optimizers import get_optimizer, SGD, Adam
from .callbacks import Callback, EarlyStopping
from .layers import Layer, Dense, iDense, jDense, BatchNorm, GnosticBatchNorm, Flatten
from .model import Model, Sequential
from .gn_activations import ActivationFunctions
from .neuron import GnosticNeuron

__all__ = [
	"Tensor",
	"History",
	"get_initializer",
	"Initializer",
	"Zeros",
	"Ones",
	"RandomNormal",
	"XavierUniform",
	"HeNormal",
	"get_activation",
	"fi",
	"fj",
	"hi",
	"hj",
	"Activation",
	"ReLU",
	"Sigmoid",
	"Tanh",
	"Softmax",
	"Fidelity",
	"Infidelity",
	"Irrelevance",
	"Relevance",
	"get_loss",
	"Loss",
	"MSE",
	"BinaryCrossEntropy",
	"GnosticFidelity",
	"GnosticInfidelity",
	"GnosticInformation",
	"GnosticResidualEntropy",
	"GnosticMSE",
	"GnosticBinaryCrossEntropy",
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
	"Callback",
	"EarlyStopping",
	"Layer",
	"Dense",
	"iDense",
	"jDense",
	"BatchNorm",
	"GnosticBatchNorm",
	"Flatten",
	"Model",
	"Sequential",
	"ActivationFunctions",
	"GnosticNeuron",
]
