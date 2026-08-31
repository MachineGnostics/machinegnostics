"""MAGNET (Machine Gnostics Neural Networks) public package exports.

Developer note
-------------
Author: Nirmal Parmar, Machine Gnostics

This is the flat user-facing MAGNET namespace. The package now runs on a
hidden PyTorch backend, but the public API stays in MAGNET terms: tensors,
layers, activations, losses, optimizers, and a small runtime configuration
entry point for device selection.

Examples
--------
>>> from machinegnostics.magnet import configure, Sequential, Dense, Sigmoid, Adam, MSE
>>> configure(device="auto")
RuntimeConfig(...)
>>> model = Sequential([Dense(2, 1), Sigmoid()])
>>> model.compile(loss=MSE(), optimizer=Adam(lr=0.01))
"""

from .core import Tensor, History, Callback, EarlyStopping, configure, get_runtime, get_torch_device, get_torch_dtype, to_numpy, to_torch, unbroadcast
from .models import Model, Sequential, GnosticNeuron
from .initializers import get_initializer, Initializer, Zeros, Ones, RandomNormal, XavierUniform, HeNormal
from .activations import (
	get_activation,
	fi,
	fj,
	hi,
	hj,
	Activation,
	ReLU,
	Step,
	LeakyReLU,
	ELU,
	Sigmoid,
	Softplus,
	Tanh,
	Swish,
	Softmax,
	Fidelity,
	Infidelity,
	Irrelevance,
	Relevance,
    GnosticProba,
    Entropy,
	FiActivation,
    Square
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
    GnosticRSS,
    GnosticISS,
	gnostic_weighted_mse,
	gnostic_weighted_rmse,
	fidelity_loss,
	infidelity_loss,
	irrelevance_loss,
	relevance_loss,
	gnostic_characteristic_loss,
)
from .optimizers import get_optimizer, Adagrad, SGD, Adam, RMSprop
from .layers import Layer, Dense, iDense, jDense, BatchNorm, GnosticBatchNorm, Flatten
from .activations.gn_activations import ActivationFunctions

__all__ = [
	"Tensor",
	"configure",
	"get_runtime",
	"get_torch_device",
	"get_torch_dtype",
	"to_numpy",
	"to_torch",
	"unbroadcast",
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
	"Step",
	"LeakyReLU",
	"ELU",
	"Sigmoid",
	"Softplus",
	"Tanh",
	"Swish",
	"Softmax",
	"Fidelity",
	"Infidelity",
	"Irrelevance",
	"Relevance",
	"FiActivation",
	FjActivation,
	"FjActivation",
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
	"Adagrad",
	"SGD",
	"Adam",
	"RMSprop",
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
	"GnosticISS",
    "GnosticRSS",
    "GnosticProba",
	"Entropy",
]
