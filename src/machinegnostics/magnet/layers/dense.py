"""Dense layers for MAGNET (Machine Gnostics Neural Networks).

Developer note
-------------
Author: Nirmal Parmar

Examples
--------
>>> import numpy as np
>>> from machinegnostics.magnet.layers.dense import Dense
>>> layer = Dense(2, 3)
>>> layer(np.array([[1.0, 2.0]])).shape
(1, 3)
"""

from __future__ import annotations

from .._gnostic import gnostic_weights_i, gnostic_weights_j
from ..initializers import XavierUniform, Zeros, get_initializer
from ..tensor import Tensor
from .base import Layer


class Dense(Layer):
	"""Fully connected linear layer with trainable weights and bias.

	The layer computes ``y = x @ W + b`` and relies on tensor autograd for the
	gradient path.
	"""

	def __init__(self, in_features, out_features, weight_init=None, bias_init=None, name=None):
		"""Create a dense layer.

		Parameters
		----------
		in_features:
			Input dimensionality.
		out_features:
			Number of output units.
		weight_init:
			Initializer for the weight matrix.
		bias_init:
			Initializer for the bias vector.
		name:
			Optional layer name.
		"""
		super().__init__(name)
		weight_init = get_initializer(weight_init) if weight_init is not None else XavierUniform()
		bias_init = get_initializer(bias_init) if bias_init is not None else Zeros()
		self.params["W"] = weight_init((in_features, out_features))
		self.params["W"].requires_grad = True
		self.params["b"] = bias_init((out_features,))
		self.params["b"].requires_grad = True
		self.grads["W"] = None
		self.grads["b"] = None

	def forward(self, x, training=True):
		"""Apply the affine transform to the input tensor."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		self.input = x
		return x @ self.params["W"] + self.params["b"]

	def backward(self, grad_output):
		"""Dense layers use tensor autograd, so manual backward is unused."""
		raise NotImplementedError("Dense uses tensor autograd; call loss.backward() instead")


class iDense(Dense):
	"""Input-weighted dense layer using gnostic estimating weights.

	This layer is useful when the tutorial needs to emphasize the input-side
	gnostic weighting path.
	"""

	def __init__(self, in_features, out_features, weight_init=None, bias_init=None, name=None, S: float | str = 2.0):
		"""Create an input-weighted dense layer."""
		super().__init__(in_features, out_features, weight_init=weight_init, bias_init=bias_init, name=name)
		self.S = S

	def forward(self, x, training=True):
		"""Apply gnostic input weights before the affine transform."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		weights = gnostic_weights_i(x.data, scale=self.S)
		if weights.shape != x.shape:
			weights = weights.reshape((1,) * (x.ndim - weights.ndim) + weights.shape)
		return super().forward(x * Tensor(weights), training=training)


class jDense(Dense):
	"""Input-weighted dense layer using gnostic quantifying weights."""

	def __init__(self, in_features, out_features, weight_init=None, bias_init=None, name=None, S: float | str = 2.0):
		"""Create a jDense layer."""
		super().__init__(in_features, out_features, weight_init=weight_init, bias_init=bias_init, name=name)
		self.S = S

	def forward(self, x, training=True):
		"""Apply gnostic quantifying weights before the affine transform."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		weights = gnostic_weights_j(x.data, scale=self.S)
		if weights.shape != x.shape:
			weights = weights.reshape((1,) * (x.ndim - weights.ndim) + weights.shape)
		return super().forward(x * Tensor(weights), training=training)
