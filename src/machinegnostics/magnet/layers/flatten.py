"""Flatten layer for MAGNET (Machine Gnostics Neural Networks).

Developer note
--------------
``Flatten`` reshapes structured inputs into a 2D batch suitable for dense
layers. It keeps the batch axis intact and records the original shape so the
layer can be understood easily when debugging a model pipeline.

Minimal working example
-----------------------
>>> import numpy as np
>>> from machinegnostics.magnet.layers.flatten import Flatten
>>> layer = Flatten()
>>> layer(np.ones((2, 3, 4))).shape
(2, 12)

Author: Nirmal Parmar
"""

from __future__ import annotations

from ..core.tensor import Tensor
from .base import Layer


class Flatten(Layer):
	"""Reshape a batch of tensors into two dimensions.

	``Flatten`` converts inputs with shape ``(batch, ...)`` into a 2D tensor of
	shape ``(batch, features)`` by collapsing all trailing dimensions into a
	single feature axis. It is the standard bridge between structured inputs and
	fully connected layers.

	Typical uses
	------------
	- image tensors before a dense classifier;
	- sequence or grid features before a linear head;
	- any model that expects a vector per sample.
	"""

	def __init__(self, name=None):
		"""Create a flatten layer.

		Parameters
		----------
		name:
			Optional display name for debugging and summaries.

		Examples
		--------
		>>> from machinegnostics.magnet import Flatten
		>>> Flatten()
		<Flatten: 0 params>
		"""
		super().__init__(name)
		self.logger.debug("Flatten initialized.")

	def forward(self, x, training=True):
		"""Flatten the trailing dimensions while preserving the batch axis.

		Parameters
		----------
		x:
			Input array or tensor with shape ``(batch, d1, d2, ...)``.
		training:
			Accepted for API consistency. Flatten behaves the same in both modes.

		Returns
		-------
		Tensor
			A tensor with shape ``(batch, -1)``.

		Examples
		--------
		>>> import numpy as np
		>>> from machinegnostics.magnet import Flatten
		>>> Flatten()(np.ones((2, 3, 4))).shape
		(2, 12)
		"""
		x = x if isinstance(x, Tensor) else Tensor(x)
		self.input_shape = x.shape
		self.logger.debug("Flattening input shape %s.", x.shape)
		return x.reshape(x.shape[0], -1)

	def backward(self, grad_output):
		"""Flatten relies on tensor autograd, so manual backward is unused.

		The layer only changes tensor shape, so autograd can propagate gradients
		through the reshape operation automatically.
		"""
		self.logger.debug("Flatten.backward called; autograd handles gradients.")
		raise NotImplementedError("Flatten uses tensor autograd; call loss.backward() instead")
