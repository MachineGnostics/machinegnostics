"""Dense layer implementations."""

from __future__ import annotations

from .._gnostic import gnostic_weights_i, gnostic_weights_j
from ..initializers import XavierUniform, Zeros, get_initializer
from ..tensor import Tensor
from .base import Layer


class Dense(Layer):
	def __init__(self, in_features, out_features, weight_init=None, bias_init=None, name=None):
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
		x = x if isinstance(x, Tensor) else Tensor(x)
		self.input = x
		return x @ self.params["W"] + self.params["b"]

	def backward(self, grad_output):
		raise NotImplementedError("Dense uses tensor autograd; call loss.backward() instead")


class iDense(Dense):
	def __init__(self, in_features, out_features, weight_init=None, bias_init=None, name=None, S: float | str = 2.0):
		super().__init__(in_features, out_features, weight_init=weight_init, bias_init=bias_init, name=name)
		self.S = S

	def forward(self, x, training=True):
		x = x if isinstance(x, Tensor) else Tensor(x)
		weights = gnostic_weights_i(x.data, scale=self.S)
		if weights.shape != x.shape:
			weights = weights.reshape((1,) * (x.ndim - weights.ndim) + weights.shape)
		return super().forward(x * Tensor(weights), training=training)


class jDense(Dense):
	def __init__(self, in_features, out_features, weight_init=None, bias_init=None, name=None, S: float | str = 2.0):
		super().__init__(in_features, out_features, weight_init=weight_init, bias_init=bias_init, name=name)
		self.S = S

	def forward(self, x, training=True):
		x = x if isinstance(x, Tensor) else Tensor(x)
		weights = gnostic_weights_j(x.data, scale=self.S)
		if weights.shape != x.shape:
			weights = weights.reshape((1,) * (x.ndim - weights.ndim) + weights.shape)
		return super().forward(x * Tensor(weights), training=training)
