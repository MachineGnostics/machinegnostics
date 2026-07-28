"""Base magnet layer class."""

from __future__ import annotations

import numpy as np


class Layer:
	def __init__(self, name=None):
		self.name = name or self.__class__.__name__
		self.params = {}
		self.grads = {}
		self.trainable = True
		self._training = True

	def forward(self, x, training=True):
		raise NotImplementedError

	def backward(self, grad_output):
		raise NotImplementedError

	def __call__(self, x, training=True):
		return self.forward(x, training=training)

	def parameters(self):
		for param in self.params.values():
			yield param

	def sync_grads(self):
		for key, param in self.params.items():
			self.grads[key] = None if param.grad is None else np.asarray(param.grad, dtype=np.float64).copy()

	def get_params_and_grads(self):
		for key, param in self.params.items():
			yield param, None if param.grad is None else np.asarray(param.grad, dtype=np.float64)

	def set_mode(self, training: bool):
		self._training = training

	def __repr__(self):
		n_params = sum(param.data.size for param in self.params.values())
		return f"<{self.name}: {n_params} params>"
