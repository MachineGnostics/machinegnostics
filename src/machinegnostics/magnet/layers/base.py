"""Base layer class for MAGNET (Machine Gnostics Neural Networks).

Developer note
-------------
Author: Nirmal Parmar
"""

from __future__ import annotations

import numpy as np


class Layer:
	"""Common base class for all MAGNET layers.

	Subclasses populate ``params`` and implement ``forward``. Trainable tensors
	are stored as ``machinegnostics.magnet.Tensor`` objects.
	"""

	def __init__(self, name=None):
		"""Initialize the layer bookkeeping fields."""
		self.name = name or self.__class__.__name__
		self.params = {}
		self.grads = {}
		self.trainable = True
		self._training = True

	def forward(self, x, training=True):
		"""Transform the input tensor in the forward pass."""
		raise NotImplementedError

	def backward(self, grad_output):
		"""Backward pass hook for non-autograd layers."""
		raise NotImplementedError

	def __call__(self, x, training=True):
		"""Alias for ``forward`` so layers can be called like functions."""
		return self.forward(x, training=training)

	def parameters(self):
		"""Yield all trainable tensors owned by the layer."""
		for param in self.params.values():
			yield param

	def sync_grads(self):
		"""Cache tensor gradients into the layer-level ``grads`` mapping."""
		for key, param in self.params.items():
			self.grads[key] = None if param.grad is None else np.asarray(param.grad, dtype=np.float64).copy()

	def get_params_and_grads(self):
		"""Yield parameter and gradient pairs for the optimizer."""
		for key, param in self.params.items():
			yield param, None if param.grad is None else np.asarray(param.grad, dtype=np.float64)

	def set_mode(self, training: bool):
		"""Record whether the layer is in training or inference mode."""
		self._training = training

	def __repr__(self):
		"""Return a concise debug representation."""
		n_params = sum(param.data.size for param in self.params.values())
		return f"<{self.name}: {n_params} params>"
