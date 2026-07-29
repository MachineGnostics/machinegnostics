"""Activation layers and gnostic characteristic helpers for MAGNET.

Developer note
-------------
Author: Nirmal Parmar

This module exposes both standard activations and the gnostic activation
family used by MAGNET (Machine Gnostics Neural Networks).

Examples
--------
>>> import numpy as np
>>> from machinegnostics.magnet.activations import Sigmoid, Fidelity
>>> Sigmoid()(np.array([-1.0, 0.0, 1.0])).shape
(3,)
"""

from __future__ import annotations

import numpy as np

from ._gnostic import compute_characteristics, custom_tensor
from .tensor import Tensor
from .layers.base import Layer


class Activation(Layer):
	"""Base class for MAGNET activation layers.

	Subclasses implement ``forward`` and rely on tensor autograd for gradients.
	"""
	def __init__(self, name=None):
		super().__init__(name)

	def forward(self, x, training=True):
		"""Transform the input tensor and return the activated output."""
		raise NotImplementedError

	def backward(self, grad_output):
		"""Activation layers use tensor autograd, so explicit backward is unused."""
		raise NotImplementedError


class ReLU(Activation):
	"""Rectified linear unit activation."""

	def forward(self, x, training=True):
		"""Return ``max(0, x)`` elementwise."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		return x.relu()


class Sigmoid(Activation):
	"""Logistic sigmoid activation."""

	def forward(self, x, training=True):
		"""Return the elementwise sigmoid of the input."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		return x.sigmoid()


class Tanh(Activation):
	"""Hyperbolic tangent activation."""

	def forward(self, x, training=True):
		"""Return the elementwise hyperbolic tangent of the input."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		return x.tanh()


class Softmax(Activation):
	"""Stable softmax activation over the last axis."""

	def forward(self, x, training=True):
		"""Convert logits to probabilities along the final dimension."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		shifted = x - Tensor(np.max(x.data, axis=-1, keepdims=True))
		exp = shifted.exp()
		return exp / exp.sum(axis=-1, keepdims=True)


def _gnostic_activation_tensor(x, value, prime):
	prime = np.asarray(prime, dtype=np.float64)
	return custom_tensor(value, [x], lambda out: x._add_grad(out.grad * prime if out.grad is not None else 0.0))


class Fidelity(Activation):
	"""Gnostic fidelity activation.

	The layer maps inputs to the gnostic fidelity characteristic and keeps the
	analytic derivative needed by the MAGNET autograd flow.
	"""
	def __init__(self, S: float | str = 1, name=None):
		"""Create a fidelity activation.

		Parameters
		----------
		S:
			Scale parameter used by the gnostic characteristic engine.
		name:
			Optional layer name.
		"""
		super().__init__(name)
		self.S = S

	def forward(self, x, training=True):
		"""Return the fidelity characteristic for the supplied tensor.

		Examples
		--------
		>>> import numpy as np
		>>> layer = Fidelity()
		>>> layer(np.array([0.1, 0.2])).shape
		(2,)
		"""
		x = x if isinstance(x, Tensor) else Tensor(x)
		info = compute_characteristics(x.data, scale=self.S)
		self.S_local = info["S_local"]
		self.fidelity = np.asarray(info["fi"], dtype=np.float64)
		self.hi = np.asarray(info["hi"], dtype=np.float64)
		prime = (-2.0 * self.fidelity * self.hi + np.finfo(float).eps) / self.S_local
		return _gnostic_activation_tensor(x, self.fidelity, prime)


class Infidelity(Activation):
	"""Gnostic infidelity activation."""

	def __init__(self, S: float | str = 1, name=None):
		"""Create an infidelity activation."""
		super().__init__(name)
		self.S = S

	def forward(self, x, training=True):
		"""Return the infidelity characteristic for the supplied tensor."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		info = compute_characteristics(x.data, scale=self.S)
		self.S_local = info["S_local"]
		self.infidelity = np.asarray(info["fj"], dtype=np.float64)
		self.hi = np.asarray(info["hi"], dtype=np.float64)
		prime = (2.0 * self.hi + np.finfo(float).eps) / (self.S_local * (self.infidelity + np.finfo(float).eps))
		return _gnostic_activation_tensor(x, self.infidelity, prime)


class Irrelevance(Activation):
	"""Gnostic irrelevance activation (``hi`` characteristic)."""

	def __init__(self, S: float | str = 1, name=None):
		"""Create an irrelevance activation."""
		super().__init__(name)
		self.S = S

	def forward(self, x, training=True):
		"""Return the irrelevance characteristic for the supplied tensor."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		info = compute_characteristics(x.data, scale=self.S)
		self.S_local = info["S_local"]
		self.irrelevance = np.asarray(info["hi"], dtype=np.float64)
		self.fi = np.asarray(info["fi"], dtype=np.float64)
		prime = (2.0 / self.S_local) * self.fi ** 2
		return _gnostic_activation_tensor(x, self.irrelevance, prime)


class Relevance(Activation):
	"""Gnostic relevance activation (``hj`` characteristic)."""

	def __init__(self, S: float | str = 1, name=None):
		"""Create a relevance activation."""
		super().__init__(name)
		self.S = S

	def forward(self, x, training=True):
		"""Return the relevance characteristic for the supplied tensor."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		info = compute_characteristics(x.data, scale=self.S)
		self.S_local = info["S_local"]
		self.relevance = np.asarray(info["hj"], dtype=np.float64)
		self.fi = np.asarray(info["fi"], dtype=np.float64)
		prime = (2.0 / self.S_local) * self.fi
		return _gnostic_activation_tensor(x, self.relevance, prime)


def fi(x, S: float | str = 1):
	"""Convenience function returning the gnostic fidelity characteristic."""
	return np.asarray(compute_characteristics(x, scale=S)["fi"], dtype=np.float64)


def fj(x, S: float | str = 1):
	"""Convenience function returning the gnostic infidelity characteristic."""
	return np.asarray(compute_characteristics(x, scale=S)["fj"], dtype=np.float64)


def hi(x, S: float | str = 1):
	"""Convenience function returning the gnostic irrelevance characteristic."""
	return np.asarray(compute_characteristics(x, scale=S)["hi"], dtype=np.float64)


def hj(x, S: float | str = 1):
	"""Convenience function returning the gnostic relevance characteristic."""
	return np.asarray(compute_characteristics(x, scale=S)["hj"], dtype=np.float64)


def get_activation(activation):
	"""Resolve a string or layer instance into an activation object.

	Examples
	--------
	>>> get_activation("relu")
	ReLU()
	"""
	if activation is None:
		return None
	if isinstance(activation, Layer):
		return activation
	if isinstance(activation, str):
		registry = {
			"relu": ReLU(),
			"sigmoid": Sigmoid(),
			"tanh": Tanh(),
			"softmax": Softmax(),
			"fidelity": Fidelity(),
			"infidelity": Infidelity(),
			"irrelevance": Irrelevance(),
			"relevance": Relevance(),
		}
		key = activation.replace("_", "").replace("-", "").lower()
		try:
			return registry[key]
		except KeyError as exc:
			raise ValueError(f"Unknown activation: {activation}") from exc
	raise TypeError(f"Unsupported activation specification: {type(activation)!r}")