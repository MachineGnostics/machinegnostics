"""Batch normalization layers for MAGNET.

Developer note
-------------
Author: Nirmal Parmar

Examples
--------
>>> import numpy as np
>>> from machinegnostics.magnet.layers.batchnorm import BatchNorm
>>> layer = BatchNorm(3)
>>> layer(np.ones((2, 3))).shape
(2, 3)
"""

from __future__ import annotations

import numpy as np

from .._gnostic import gnostic_weights_i
from ..tensor import Tensor
from .base import Layer


class BatchNorm(Layer):
	"""Standard batch normalization layer for feature vectors.

	The layer maintains running mean and variance for inference-time use.
	"""

	def __init__(self, num_features, momentum=0.9, eps=1e-5, name=None):
		"""Create a batch-normalization layer."""
		super().__init__(name)
		self.momentum = momentum
		self.eps = eps
		self.params["gamma"] = Tensor(np.ones(num_features, dtype=np.float64), requires_grad=True)
		self.params["beta"] = Tensor(np.zeros(num_features, dtype=np.float64), requires_grad=True)
		self.grads["gamma"] = None
		self.grads["beta"] = None
		self.running_mean = np.zeros(num_features, dtype=np.float64)
		self.running_var = np.ones(num_features, dtype=np.float64)

	def forward(self, x, training=True):
		"""Normalize the batch during training and reuse running stats at inference."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		if training:
			batch_mean = x.data.mean(axis=0)
			batch_var = x.data.var(axis=0)
			self.centered = x - Tensor(batch_mean)
			self.std_inv = Tensor(1.0 / np.sqrt(batch_var + self.eps))
			self.x_norm = self.centered * self.std_inv
			self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
			self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
		else:
			self.x_norm = (x - Tensor(self.running_mean)) / Tensor(np.sqrt(self.running_var + self.eps))
		return self.params["gamma"] * self.x_norm + self.params["beta"]

	def backward(self, grad_output):
		"""BatchNorm uses tensor autograd, so explicit backward is unused."""
		raise NotImplementedError("BatchNorm uses tensor autograd; call loss.backward() instead")


class GnosticBatchNorm(BatchNorm):
	"""Batch normalization variant that applies gnostic sample weighting."""

	def forward(self, x, training=True):
		"""Normalize the batch with an additional gnostic weighting term."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		if training:
			batch_mean = x.data.mean(axis=0)
			batch_var = x.data.var(axis=0)
			self.centered = x - Tensor(batch_mean)
			self.gw = Tensor(gnostic_weights_i(self.centered.data, scale=2.0))
			self.std_inv = Tensor(1.0 / np.sqrt(batch_var + self.eps))
			self.x_norm = self.centered * self.std_inv * self.gw
			self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
			self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
		else:
			self.x_norm = (x - Tensor(self.running_mean)) / Tensor(np.sqrt(self.running_var + self.eps))
		return self.params["gamma"] * self.x_norm + self.params["beta"]
