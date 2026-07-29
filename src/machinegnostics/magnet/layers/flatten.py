"""Flatten layer for MAGNET (Machine Gnostics Neural Networks).

Developer note
-------------
Author: Nirmal Parmar
"""

from __future__ import annotations

from ..tensor import Tensor
from .base import Layer


class Flatten(Layer):
	"""Reshape a batch of tensors into two dimensions.

	This is useful when moving from structured inputs to a dense classifier.
	"""

	def forward(self, x, training=True):
		"""Flatten the trailing dimensions while preserving the batch axis."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		self.input_shape = x.shape
		return x.reshape(x.shape[0], -1)

	def backward(self, grad_output):
		"""Flatten relies on tensor autograd, so manual backward is unused."""
		raise NotImplementedError("Flatten uses tensor autograd; call loss.backward() instead")
