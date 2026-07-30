"""Batch normalization layers for MAGNET.

Developer note
--------------
``BatchNorm`` keeps running statistics for inference and normalizes feature
vectors during training. ``GnosticBatchNorm`` extends that flow by applying a
gnostic weighting term before the normalized output is produced.

Minimal working example
-----------------------
>>> import numpy as np
>>> from machinegnostics.magnet import BatchNorm
>>> layer = BatchNorm(3)
>>> layer(np.ones((2, 3))).shape
(2, 3)

Author: Nirmal Parmar
"""

from __future__ import annotations

import numpy as np

from ..core._gnostic import gnostic_weights_i, gnostic_weights_j
from ..core.tensor import Tensor
from .base import Layer


class BatchNorm(Layer):
	"""Standard batch normalization layer for feature vectors.

	The layer normalizes each feature dimension using batch statistics during
	training and running statistics during inference. This makes optimization
	more stable by reducing internal feature-scale drift.

	The layer stores four important pieces of state:
	- ``gamma``: trainable scale parameter;
	- ``beta``: trainable shift parameter;
	- ``running_mean``: moving average used for inference;
	- ``running_var``: moving variance used for inference.

	Examples
	--------
	>>> import numpy as np
	>>> from machinegnostics.magnet import BatchNorm
	>>> layer = BatchNorm(3)
	>>> layer(np.ones((2, 3))).shape
	(2, 3)
	"""

	def __init__(self, num_features, momentum=0.9, eps=1e-5, name=None):
		"""Create a batch-normalization layer.

		Parameters
		----------
		num_features:
			Number of features in the last dimension of the input.
		momentum:
			Update rate for the running statistics.
		eps:
			Small constant added to the variance for numerical stability.
		name:
			Optional layer name.

		Examples
		--------
		>>> from machinegnostics.magnet import BatchNorm
		>>> BatchNorm(4)
		<BatchNorm: 8 params>
		"""
		super().__init__(name)
		self.momentum = momentum
		self.eps = eps
		self.params["gamma"] = Tensor(np.ones(num_features, dtype=np.float64), requires_grad=True)
		self.params["beta"] = Tensor(np.zeros(num_features, dtype=np.float64), requires_grad=True)
		self.grads["gamma"] = None
		self.grads["beta"] = None
		self.running_mean = np.zeros(num_features, dtype=np.float64)
		self.running_var = np.ones(num_features, dtype=np.float64)
		self.logger.debug("BatchNorm initialized with num_features=%s.", num_features)

	def forward(self, x, training=True):
		"""Normalize the batch during training and reuse running stats at inference.

		During training, the layer computes the mean and variance from the current
		mini-batch, normalizes the input, and updates the running statistics.
		During inference, it skips batch statistics and reuses the accumulated
		running values so predictions are deterministic.

		Parameters
		----------
		x:
			Input array or tensor of shape ``(batch, num_features)``.
		training:
			If ``True``, use batch statistics and update running statistics. If
			``False``, use the stored running statistics.

		Returns
		-------
		Tensor
			The normalized and affine-transformed output.
		"""
		x = x if isinstance(x, Tensor) else Tensor(x)
		if training:
			self.logger.debug("BatchNorm forward in training mode with input shape %s.", x.shape)
			batch_mean = x.data.mean(axis=0)
			batch_var = x.data.var(axis=0)
			self.centered = x - Tensor(batch_mean)
			self.std_inv = Tensor(1.0 / np.sqrt(batch_var + self.eps))
			self.x_norm = self.centered * self.std_inv
			self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
			self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
		else:
			self.logger.debug("BatchNorm forward in inference mode with input shape %s.", x.shape)
			self.x_norm = (x - Tensor(self.running_mean)) / Tensor(np.sqrt(self.running_var + self.eps))
		return self.params["gamma"] * self.x_norm + self.params["beta"]

	def backward(self, grad_output):
		"""BatchNorm uses tensor autograd, so explicit backward is unused.

		The normalization graph is built from differentiable tensor operations,
		so gradients are computed automatically when the final loss is backpropagated.
		"""
		self.logger.debug("BatchNorm.backward called; autograd handles gradients.")
		raise NotImplementedError("BatchNorm uses tensor autograd; call loss.backward() instead")


class GnosticBatchNorm(BatchNorm):
	"""Batch normalization variant that applies gnostic sample weighting.

	``GnosticBatchNorm`` extends :class:`BatchNorm` by computing a gnostic
	weighting tensor from the centered batch and multiplying it into the normal-
	ized activations. The ``kind`` flag selects the weighting family:

	- ``"i"`` uses estimating weights;
	- ``"j"`` uses quantifying weights.

	Examples
	--------
	>>> import numpy as np
	>>> from machinegnostics.magnet import GnosticBatchNorm
	>>> layer = GnosticBatchNorm(3, kind="i")
	>>> layer(np.ones((2, 3))).shape
	(2, 3)
	"""
	def __init__(self, num_features, momentum=0.9, eps=1e-5, name=None, S:float|str=2.0, kind:str="i"):
		"""Create a gnostic batch-normalization layer.

		Parameters
		----------
		num_features:
			Number of features in the last dimension of the input.
		momentum:
			Update rate for running statistics.
		eps:
			Small constant for numerical stability.
		name:
			Optional layer name.
		S:
			Scale parameter passed to the gnostic weighting helper.
		kind:
			Weighting family to use, either ``"i"`` or ``"j"``.
		"""
		super().__init__(num_features, momentum, eps, name)
		self.S = S
		self.kind = kind
		self.logger.debug("GnosticBatchNorm initialized with kind=%s, scale=%s.", kind, S)

	def forward(self, x, training=True):
		"""Normalize the batch with an additional gnostic weighting term.

		The training-time flow is:
		1. compute batch mean and variance;
		2. center the batch;
		3. derive a gnostic weighting tensor from the centered data;
		4. normalize and weight the activations;
		5. update the running statistics.

		At inference time, the layer reuses the running statistics and skips the
		gnostic weighting step so the output remains deterministic.
		"""
		x = x if isinstance(x, Tensor) else Tensor(x)
		if training:
			self.logger.debug("GnosticBatchNorm forward in training mode with input shape %s.", x.shape)
			batch_mean = x.data.mean(axis=0)
			batch_var = x.data.var(axis=0)
			self.centered = x - Tensor(batch_mean)
			if self.kind == "i":
				self.gw = Tensor(gnostic_weights_i(self.centered.data, scale=self.S))
			elif self.kind == "j":
				self.gw = Tensor(gnostic_weights_j(self.centered.data, scale=self.S))
			else:
				raise ValueError(f"Unsupported kind: {self.kind}")
			self.std_inv = Tensor(1.0 / np.sqrt(batch_var + self.eps))
			self.x_norm = self.centered * self.std_inv * self.gw
			self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
			self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
		else:
			self.logger.debug("GnosticBatchNorm forward in inference mode with input shape %s.", x.shape)
			self.x_norm = (x - Tensor(self.running_mean)) / Tensor(np.sqrt(self.running_var + self.eps))
		return self.params["gamma"] * self.x_norm + self.params["beta"]
