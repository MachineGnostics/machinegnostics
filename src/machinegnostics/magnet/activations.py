"""Activation layers and gnostic characteristic helpers."""

from __future__ import annotations

import numpy as np

from ._gnostic import compute_characteristics, custom_tensor
from .tensor import Tensor
from .layers.base import Layer


class Activation(Layer):
	def __init__(self, name=None):
		super().__init__(name)

	def forward(self, x, training=True):
		raise NotImplementedError

	def backward(self, grad_output):
		raise NotImplementedError


class ReLU(Activation):
	def forward(self, x, training=True):
		x = x if isinstance(x, Tensor) else Tensor(x)
		return x.relu()


class Sigmoid(Activation):
	def forward(self, x, training=True):
		x = x if isinstance(x, Tensor) else Tensor(x)
		return x.sigmoid()


class Tanh(Activation):
	def forward(self, x, training=True):
		x = x if isinstance(x, Tensor) else Tensor(x)
		return x.tanh()


class Softmax(Activation):
	def forward(self, x, training=True):
		x = x if isinstance(x, Tensor) else Tensor(x)
		shifted = x - Tensor(np.max(x.data, axis=-1, keepdims=True))
		exp = shifted.exp()
		return exp / exp.sum(axis=-1, keepdims=True)


def _gnostic_activation_tensor(x, value, prime):
	prime = np.asarray(prime, dtype=np.float64)
	return custom_tensor(value, [x], lambda out: x._add_grad(out.grad * prime if out.grad is not None else 0.0))


class Fidelity(Activation):
	def __init__(self, S: float | str = 1, name=None):
		super().__init__(name)
		self.S = S

	def forward(self, x, training=True):
		x = x if isinstance(x, Tensor) else Tensor(x)
		info = compute_characteristics(x.data, scale=self.S)
		self.S_local = info["S_local"]
		self.fidelity = np.asarray(info["fi"], dtype=np.float64)
		self.hi = np.asarray(info["hi"], dtype=np.float64)
		prime = (-2.0 * self.fidelity * self.hi + np.finfo(float).eps) / self.S_local
		return _gnostic_activation_tensor(x, self.fidelity, prime)


class Infidelity(Activation):
	def __init__(self, S: float | str = 1, name=None):
		super().__init__(name)
		self.S = S

	def forward(self, x, training=True):
		x = x if isinstance(x, Tensor) else Tensor(x)
		info = compute_characteristics(x.data, scale=self.S)
		self.S_local = info["S_local"]
		self.infidelity = np.asarray(info["fj"], dtype=np.float64)
		self.hi = np.asarray(info["hi"], dtype=np.float64)
		prime = (2.0 * self.hi + np.finfo(float).eps) / (self.S_local * (self.infidelity + np.finfo(float).eps))
		return _gnostic_activation_tensor(x, self.infidelity, prime)


class Irrelevance(Activation):
	def __init__(self, S: float | str = 1, name=None):
		super().__init__(name)
		self.S = S

	def forward(self, x, training=True):
		x = x if isinstance(x, Tensor) else Tensor(x)
		info = compute_characteristics(x.data, scale=self.S)
		self.S_local = info["S_local"]
		self.irrelevance = np.asarray(info["hi"], dtype=np.float64)
		self.fi = np.asarray(info["fi"], dtype=np.float64)
		prime = (2.0 / self.S_local) * self.fi ** 2
		return _gnostic_activation_tensor(x, self.irrelevance, prime)


class Relevance(Activation):
	def __init__(self, S: float | str = 1, name=None):
		super().__init__(name)
		self.S = S

	def forward(self, x, training=True):
		x = x if isinstance(x, Tensor) else Tensor(x)
		info = compute_characteristics(x.data, scale=self.S)
		self.S_local = info["S_local"]
		self.relevance = np.asarray(info["hj"], dtype=np.float64)
		self.fi = np.asarray(info["fi"], dtype=np.float64)
		prime = (2.0 / self.S_local) * self.fi
		return _gnostic_activation_tensor(x, self.relevance, prime)


def fi(x, S: float | str = 1):
	return np.asarray(compute_characteristics(x, scale=S)["fi"], dtype=np.float64)


def fj(x, S: float | str = 1):
	return np.asarray(compute_characteristics(x, scale=S)["fj"], dtype=np.float64)


def hi(x, S: float | str = 1):
	return np.asarray(compute_characteristics(x, scale=S)["hi"], dtype=np.float64)


def hj(x, S: float | str = 1):
	return np.asarray(compute_characteristics(x, scale=S)["hj"], dtype=np.float64)


def get_activation(activation):
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