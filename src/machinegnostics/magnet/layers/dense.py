"""Dense layer for MAGNET (Machine Gnostics Neural Networks).

Developer note
--------------
``Dense`` is the canonical fully connected layer used throughout MAGNET.
It owns two trainable tensors, ``W`` and ``b``, and relies on tensor autograd
for the backward pass. The layer inherits the shared logging and mode handling
from :class:`~machinegnostics.magnet.layers.base.Layer`, so it can be used in
isolation or inside a ``Model``/``Sequential`` container.

Minimal working example
-----------------------
>>> import numpy as np
>>> from machinegnostics.magnet.layers.dense import Dense
>>> layer = Dense(2, 3)
>>> layer(np.array([[1.0, 2.0]])).shape
(1, 3)

Author: Nirmal Parmar
"""

from __future__ import annotations

from ..initializers import XavierUniform, Zeros, get_initializer
from ..core.tensor import Tensor
from .base import Layer


class Dense(Layer):
	"""Fully connected linear layer with trainable weights and bias.

	The layer computes the affine transform ``y = x @ W + b``. It is the core
	building block for most MAGNET networks because it turns input features into
	a learned representation using a trainable weight matrix and bias vector.

	Unlike a standalone NumPy implementation, ``Dense`` stores its parameters as
	``Tensor`` objects so gradients can flow through MAGNET's autograd engine.
	The layer itself does not implement a manual backward pass because the
	tensor system already knows how to differentiate matrix multiplication and
	addition.

	Typical uses
	------------
	- binary or multiclass classifiers;
	- regression heads;
	- the linear part of a larger gnostic network.
	"""

	def __init__(self, in_features, out_features, weight_init=None, bias_init=None, name=None, verbose: bool = False):
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
		verbose:
			Enable debug logging for the layer instance.

		Examples
		--------
		>>> import numpy as np
		>>> from machinegnostics.magnet import Dense
		>>> layer = Dense(2, 1)
		>>> layer(np.array([[1.0, 2.0]])).shape
		(1, 1)
		"""
		super().__init__(name, verbose=verbose)
		weight_init = get_initializer(weight_init) if weight_init is not None else XavierUniform()
		bias_init = get_initializer(bias_init) if bias_init is not None else Zeros()
		self.params["W"] = weight_init((in_features, out_features))
		self.params["W"].requires_grad = True
		self.params["b"] = bias_init((out_features,))
		self.params["b"].requires_grad = True
		self.grads["W"] = None
		self.grads["b"] = None
		self.logger.debug("Dense initialized with in_features=%s, out_features=%s.", in_features, out_features)

	def forward(self, x, training=True):
		"""Apply the affine transform to the input tensor.

		Parameters
		----------
		x:
			Input array or tensor.
		training:
			Ignored by the dense computation but accepted for API consistency.

		Returns
		-------
		Tensor
			The affine output ``x @ W + b``.

		Notes
		-----
		The layer caches the input tensor on ``self.input`` so that debugging or
		inspection code can inspect the last forward pass.
		"""
		x = x if isinstance(x, Tensor) else Tensor(x)
		self.input = x
		self.logger.debug("Running dense forward pass with input shape %s.", x.shape)
		return x @ self.params["W"] + self.params["b"]

	def backward(self, grad_output):
		"""Dense layers use tensor autograd, so manual backward is unused.

		The ``grad_output`` argument is accepted only to keep the API aligned
		with non-autograd layers. Calling this method directly is not part of the
		normal MAGNET workflow; gradients are produced by ``loss.backward()`` on
		the final scalar loss instead.
		"""
		self.logger.debug("Dense.backward called; autograd handles gradients.")
		raise NotImplementedError("Dense uses tensor autograd; call loss.backward() instead")
