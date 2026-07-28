"""Flatten layer."""

from __future__ import annotations

from ..tensor import Tensor
from .base import Layer


class Flatten(Layer):
	def forward(self, x, training=True):
		x = x if isinstance(x, Tensor) else Tensor(x)
		self.input_shape = x.shape
		return x.reshape(x.shape[0], -1)

	def backward(self, grad_output):
		raise NotImplementedError("Flatten uses tensor autograd; call loss.backward() instead")
