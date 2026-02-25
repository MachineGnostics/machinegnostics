"""Activation namespace for Magnet.

Author: Nirmal Parmar
"""

from .activations import (
	GELU,
	Identity,
	LeakyReLU,
	ReLU,
	Sigmoid,
	Softmax,
	Tanh,
	gelu,
	leaky_relu,
	relu,
	sigmoid,
	softmax,
	tanh,
)

__all__ = [
	"ReLU",
	"LeakyReLU",
	"Sigmoid",
	"Tanh",
	"Softmax",
	"GELU",
	"Identity",
	"relu",
	"leaky_relu",
	"sigmoid",
	"tanh",
	"softmax",
	"gelu",
]
