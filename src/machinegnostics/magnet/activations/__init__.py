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
import torch

from ..core._gnostic import compute_characteristics, custom_tensor
from ..core.tensor import Tensor
from ..layers.base import Layer


class Activation(Layer):
	"""Base class for MAGNET activation layers.

	This class defines the shared interface for all standard and gnostic
	activations in MAGNET. Subclasses implement ``forward`` and return a tensor
	that participates in the library's autograd flow.

	Examples
	--------
	>>> class DoubleActivation(Activation):
	... 	def forward(self, x, training=True):
	... 		return x if isinstance(x, Tensor) else Tensor(x)
	>>> isinstance(DoubleActivation(), Activation)
	True
	"""
	def __init__(self, name=None, verbose: bool = False):
		super().__init__(name, verbose=verbose)

	def forward(self, x, training=True):
		"""Transform the input tensor and return the activated output."""
		raise NotImplementedError

	def backward(self, grad_output):
		"""Activation layers use tensor autograd, so explicit backward is unused."""
		raise NotImplementedError


class ReLU(Activation):
	"""Rectified linear unit activation.

	ReLU returns zero for negative inputs and preserves positive values.

	Examples
	--------
	>>> import numpy as np
	>>> from machinegnostics.magnet import Dense, ReLU, Sequential
	>>> model = Sequential([Dense(2, 2), ReLU()])
	>>> model(np.array([[1.0, -2.0]])).shape
	(1, 2)
	"""

	def forward(self, x, training=True):
		"""Return ``max(0, x)`` elementwise."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		return x.relu()


class Sigmoid(Activation):
	"""Logistic sigmoid activation.

	Sigmoid maps values into the open interval $(0, 1)$ and is commonly used
	for binary classification output layers.

	Examples
	--------
	>>> import numpy as np
	>>> from machinegnostics.magnet import Dense, Sequential, Sigmoid
	>>> model = Sequential([Dense(2, 1), Sigmoid()])
	>>> model(np.array([[0.0, 0.0]])).shape
	(1, 1)
	"""

	def forward(self, x, training=True):
		"""Return the elementwise sigmoid of the input."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		return x.sigmoid()


class Tanh(Activation):
	"""Hyperbolic tangent activation.

	Tanh squashes inputs into $(-1, 1)$ and is often used in hidden layers.

	Examples
	--------
	>>> import numpy as np
	>>> from machinegnostics.magnet import Dense, Sequential, Tanh
	>>> model = Sequential([Dense(3, 2), Tanh()])
	>>> model(np.array([[1.0, 0.0, -1.0]])).shape
	(1, 2)
	"""

	def forward(self, x, training=True):
		"""Return the elementwise hyperbolic tangent of the input."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		return x.tanh()


class Softmax(Activation):
	"""Stable softmax activation over the last axis.

	Softmax converts logits into probabilities that sum to 1 along the final
	dimension.

	Examples
	--------
	>>> import numpy as np
	>>> from machinegnostics.magnet import Dense, Sequential, Softmax
	>>> model = Sequential([Dense(3, 3), Softmax()])
	>>> model(np.array([[1.0, 2.0, 3.0]])).shape
	(1, 3)
	"""

	def forward(self, x, training=True):
		"""Convert logits to probabilities along the final dimension."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		shifted = x - Tensor(np.max(x.data, axis=-1, keepdims=True))
		exp = shifted.exp()
		return exp / exp.sum(axis=-1, keepdims=True)


class Step(Activation):
	"""Hard step activation that maps values to 0 or 1.

	Step is useful for thresholding and discrete gating. It is not smooth, so
	it should be used when a binary output is more important than gradient-rich
	training dynamics.

	Examples
	--------
	>>> import numpy as np
	>>> from machinegnostics.magnet import Dense, Sequential, Step
	>>> model = Sequential([Dense(3, 2), Step()])
	>>> model(np.array([[0.0, 1.0, -1.0]])).shape
	(1, 2)
	"""

	def __init__(self, threshold: float = 0.0, name=None, verbose: bool = False):
		"""Create a step activation."""
		super().__init__(name, verbose=verbose)
		self.threshold = float(threshold)

	def forward(self, x, training=True):
		"""Return 1 where the input exceeds the threshold, else 0."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		data = (x.data > self.threshold).astype(np.float64)
		prime = np.zeros_like(x.data, dtype=np.float64)
		return _gnostic_activation_tensor(x, data, prime)


class LeakyReLU(Activation):
	"""Leaky rectified linear unit activation.

	LeakyReLU keeps a small slope for negative inputs so gradients can flow even
	when activations fall below zero.

	Examples
	--------
	>>> import numpy as np
	>>> from machinegnostics.magnet import Dense, LeakyReLU, Sequential
	>>> model = Sequential([Dense(2, 2), LeakyReLU(alpha=0.1)])
	>>> model(np.array([[-2.0, 3.0]])).shape
	(1, 2)
	"""

	def __init__(self, alpha: float = 0.01, name=None, verbose: bool = False):
		"""Create a leaky ReLU activation."""
		super().__init__(name, verbose=verbose)
		self.alpha = float(alpha)

	def forward(self, x, training=True):
		"""Return ``x`` for positive inputs and ``alpha * x`` otherwise."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		data = np.where(x.data > 0, x.data, self.alpha * x.data)
		prime = np.where(x.data > 0, 1.0, self.alpha)
		return _gnostic_activation_tensor(x, data, prime)


class ELU(Activation):
	"""Exponential linear unit activation.

	ELU behaves like an identity for positive inputs and transitions smoothly to
	a negative exponential for values below zero.

	Examples
	--------
	>>> import numpy as np
	>>> from machinegnostics.magnet import Dense, ELU, Sequential
	>>> model = Sequential([Dense(2, 2), ELU(alpha=1.0)])
	>>> model(np.array([[1.0, -1.0]])).shape
	(1, 2)
	"""

	def __init__(self, alpha: float = 1.0, name=None, verbose: bool = False):
		"""Create an ELU activation."""
		super().__init__(name, verbose=verbose)
		self.alpha = float(alpha)

	def forward(self, x, training=True):
		"""Return the elementwise ELU transform."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		positive = x.data > 0
		data = np.where(positive, x.data, self.alpha * (np.expm1(x.data)))
		prime = np.where(positive, 1.0, data + self.alpha)
		return _gnostic_activation_tensor(x, data, prime)


class Softplus(Activation):
	"""Softplus activation.

	Softplus is a smooth approximation of ReLU and is useful when a strictly
	positive, differentiable output is desired.

	Examples
	--------
	>>> import numpy as np
	>>> from machinegnostics.magnet import Dense, Sequential, Softplus
	>>> model = Sequential([Dense(2, 1), Softplus()])
	>>> model(np.array([[0.0, 0.0]])).shape
	(1, 1)
	"""

	def forward(self, x, training=True):
		"""Return a smooth approximation of ReLU."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		data = np.log1p(np.exp(-np.abs(x.data))) + np.maximum(x.data, 0)
		prime = 1.0 / (1.0 + np.exp(-np.clip(x.data, -500, 500)))
		return _gnostic_activation_tensor(x, data, prime)


class Swish(Activation):
	"""Swish activation, defined as ``x * sigmoid(x)``.

	Swish is a smooth, self-gated activation that often performs well as a
	drop-in alternative to ReLU in deep networks.

	Examples
	--------
	>>> import numpy as np
	>>> from machinegnostics.magnet import Dense, Sequential, Swish
	>>> model = Sequential([Dense(2, 2), Swish()])
	>>> model(np.array([[0.0, 1.0]])).shape
	(1, 2)
	"""

	def forward(self, x, training=True):
		"""Return the elementwise swish transform."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		sigmoid = 1.0 / (1.0 + np.exp(-np.clip(x.data, -500, 500)))
		data = x.data * sigmoid
		prime = sigmoid + x.data * sigmoid * (1.0 - sigmoid)
		return _gnostic_activation_tensor(x, data, prime)


def _gnostic_activation_tensor(x, value, prime):
	prime = np.asarray(prime, dtype=np.float64)
	return custom_tensor(value, x, prime)


class Fidelity(Activation):
	"""Gnostic fidelity activation.

	The layer maps inputs to the gnostic fidelity characteristic and keeps the
	analytic derivative needed by the MAGNET autograd flow.

	Examples
	--------
	>>> import numpy as np
	>>> from machinegnostics.magnet import Dense, Fidelity, Sequential
	>>> model = Sequential([Dense(2, 2), Fidelity()])
	>>> model(np.array([[0.1, 0.2]])).shape
	(1, 2)
	"""
	def __init__(self, S: float | str = 1, name=None, verbose: bool = False):
		"""Create a fidelity activation.

		Parameters
		----------
		S:
			Scale parameter used by the gnostic characteristic engine.
		name:
			Optional layer name.
		"""
		super().__init__(name, verbose=verbose)
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
		prime = -(2.0 * self.fidelity * self.hi + np.finfo(float).eps) / self.S_local
		return _gnostic_activation_tensor(x, self.fidelity, prime)


class FiActivation(Activation):
	"""Trainable fidelity activation with learnable center and scale.

	FiActivation implements the normalized deviation used by the gnostic
	FiDense prototype, but as a standalone activation that can be composed with
	any MAGNET dense layer. It learns a concept center ``z0`` and a bounded
	scale ``S`` while returning the fidelity response
	``sech(2 * ((z - z0) / S))``.

	Examples
	--------
	>>> import numpy as np
	>>> from machinegnostics.magnet import Dense, FiActivation, Sequential
	>>> model = Sequential([Dense(2, 1), FiActivation()])
	>>> model(np.array([[0.0, 1.0]])).shape
	(1, 1)
	"""
	
	def __init__(self, z0_init: float | str = "mean", S_init: float = 1.0, name=None, verbose: bool = False):
		"""Create a trainable fidelity activation.
		Parameters
		----------
		z0_init:
			Initial value for the concept center. Can be a float, "mean", or "median".
		S_init:
			Initial value for the scale parameter.
		name:
			Optional layer name.
		verbose:
			Enable verbose output during training.
		"""
		super().__init__(name, verbose=verbose)
		self.z0_init = z0_init
		self.S_init = float(S_init)
		self._initialized = False

	@staticmethod
	def _stable_sech(x: torch.Tensor) -> torch.Tensor:
		abs_x = torch.abs(x)
		return torch.where(abs_x <= 20.0, 1.0 / torch.cosh(torch.clamp(x, -20.0, 20.0)), 2.0 * torch.exp(-abs_x))

	def _initialize_params(self, x: Tensor) -> None:
		if x.ndim == 0:
			feature_shape = ()
		elif x.ndim == 1:
			feature_shape = (x.shape[0],)
		else:
			feature_shape = tuple(x.shape[1:])

		if self.z0_init == "mean":
			z0_value = np.mean(x.data, axis=0, keepdims=x.ndim > 1)
		elif self.z0_init == "median":
			z0_value = np.median(x.data, axis=0, keepdims=x.ndim > 1)
		else:
			z0_value = np.full(feature_shape or (1,), float(self.z0_init), dtype=np.float64)

		s_init = float(np.clip(self.S_init, 1e-4, 1.9999))
		s_raw_value = np.full(np.shape(z0_value) or (1,), np.log(s_init / (2.0 - s_init)), dtype=np.float64)

		self.params["z0"] = Tensor(z0_value, requires_grad=True)
		self.params["S_raw"] = Tensor(s_raw_value, requires_grad=True)
		self.params["S"] = Tensor(np.full(np.shape(z0_value) or (1,), s_init, dtype=np.float64), requires_grad=False)
		self.grads["z0"] = None
		self.grads["S_raw"] = None
		self.grads["S"] = None
		self._initialized = True

	def forward(self, x, training=True):
		"""Return the fidelity characteristic for the supplied tensor.
		
		Examples
		--------
		>>> import numpy as np
		>>> from machinegnostics.magnet import Dense, FiActivation, Sequential
		>>> model = Sequential([Dense(2, 1), FiActivation()])
		>>> model(np.array([[0.0, 1.0]])).shape
		(1, 1)
		"""
		x = x if isinstance(x, Tensor) else Tensor(x)
		if not self._initialized:
			self._initialize_params(x)

		z0 = self.params["z0"]._tensor
		s_raw = self.params["S_raw"]._tensor
		s = torch.clamp(2.0 * torch.sigmoid(s_raw), 1e-4, 1.9999)
		self.params["S"] = Tensor.from_torch(s.detach().clone())

		theta = (x._tensor - z0) / s
		out = self._stable_sech(2.0 * theta)
		self.theta = Tensor.from_torch(theta)
		self.out = Tensor.from_torch(out)
		return Tensor.from_torch(out)


class Infidelity(Activation):
	"""Gnostic infidelity activation.

	This activation returns the gnostic infidelity characteristic and is useful
	when the model should emphasize the complementary characteristic to
	fidelity.

	Examples
	--------
	>>> import numpy as np
	>>> from machinegnostics.magnet import Dense, Infidelity, Sequential
	>>> model = Sequential([Dense(2, 2), Infidelity()])
	>>> model(np.array([[0.1, 0.2]])).shape
	(1, 2)
	"""

	def __init__(self, S: float | str = 1, name=None, verbose: bool = False):
		"""Create an infidelity activation."""
		super().__init__(name, verbose=verbose)
		self.S = S

	def forward(self, x, training=True):
		"""Return the infidelity characteristic for the supplied tensor."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		info = compute_characteristics(x.data, scale=self.S)
		self.S_local = info["S_local"]
		self.infidelity = np.asarray(info["fj"], dtype=np.float64)
		self.hi = np.asarray(info["hi"], dtype=np.float64)
		prime = (2 / self.S_local) * self.infidelity * self.hi
		return _gnostic_activation_tensor(x, self.infidelity, prime)


class Irrelevance(Activation):
	"""Gnostic irrelevance activation (``hj`` characteristic).

	Irrelevance captures the gnostic quantifying ``hj`` characteristic and can be used when
	the model needs a direct measure of irrelevance.

	Examples
	--------
	>>> import numpy as np
	>>> from machinegnostics.magnet import Dense, Irrelevance, Sequential
	>>> model = Sequential([Dense(2, 2), Irrelevance()])
	>>> model(np.array([[0.1, 0.2]])).shape
	(1, 2)
	"""

	def __init__(self, S: float | str = 1, name=None, verbose: bool = False):
		"""Create an irrelevance activation."""
		super().__init__(name, verbose=verbose)
		self.S = S

	def forward(self, x, training=True):
		"""Return the irrelevance characteristic for the supplied tensor."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		info = compute_characteristics(x.data, scale=self.S)
		self.S_local = info["S_local"]
		self.irrelevance = (np.asarray(info["hj"], dtype=np.float64))
		prime = (2.0 / self.S_local) * (1 - self.irrelevance ** 2)
		return _gnostic_activation_tensor(x, self.irrelevance, prime)

class GnosticProba(Activation):
	"""Gnostic probability activation.

	This activation returns the gnostic probability characteristic and is useful
	when the model should emphasize the complementary characteristic to relevance.

	Examples
	--------
	>>> import numpy as np
	>>> from machinegnostics.magnet import Dense, GnosticProba, Sequential
	>>> model = Sequential([Dense(2, 2), GnosticProba()])
	>>> model(np.array([[0.1, 0.2]])).shape
	(1, 2)
	"""

	def __init__(self, S: float | str = 1, 
			  name=None, 
			  case:str = "i", # i or j
			  verbose: bool = False):
		"""Create a gnostic probability activation."""
		super().__init__(name, verbose=verbose)
		self.S = S
		self.case = case

	def forward(self, x, training=True):
		"""Return the gnostic probability characteristic for the supplied tensor."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		info = compute_characteristics(x.data, scale=self.S)
		self.S_local = info["S_local"]
		self.char = info['characteristics']
		self.h = np.asarray(info["hi"], dtype=np.float64)
		self.proba = self.char._idistfun(self.h)
		prime = - (4 / self.S_local) * (1 - self.proba) * self.proba
		return _gnostic_activation_tensor(x, self.proba, prime)

class Entropy(Activation):
	"""Gnostic entropy activation.

	This activation returns the gnostic entropy characteristic and is useful
	when the model should emphasize the complementary characteristic to
	fidelity.

	Examples
	--------
	>>> import numpy as np
	>>> from machinegnostics.magnet import Dense, Entropy, Sequential
	>>> model = Sequential([Dense(2, 2), Entropy()])
	>>> model(np.array([[0.1, 0.2]])).shape
	(1, 2)
	"""

	def __init__(self, S: float | str = 1, 
			  name=None, 
			  case:str = "i", # i or j
			  verbose: bool = False):
		"""Create a gnostic entropy activation."""
		super().__init__(name, verbose=verbose)
		self.S = S
		self.case = case

	def forward(self, x, training=True):
		"""Return the gnostic entropy characteristic for the supplied tensor."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		info = compute_characteristics(x.data, scale=self.S)
		self.S_local = info["S_local"]
		self.char = info['characteristics']
		if self.case == "i":
			self.fi = np.asarray(info["fi"], dtype=np.float64)
			self.hi = np.asarray(info["hi"], dtype=np.float64)
			self.entropy = self.char._ientropy(self.fi)
			prime = (2 / self.S_local) * self.fi * self.hi
		elif self.case == "j":
			self.fj = np.asarray(info["fj"], dtype=np.float64)
			self.hi = np.asarray(info["hi"], dtype=np.float64)
			self.entropy = self.char._jentropy(self.fj)
			prime = (2 / self.S_local) * self.fj * self.hi
		else:
			raise ValueError(f"Invalid case: {self.case}. Must be 'i' or 'j'.")
		return _gnostic_activation_tensor(x, self.entropy, prime)


class Relevance(Activation):
	"""Gnostic relevance activation (``hi`` characteristic).

	Relevance captures the gnostic estimating Relevance (``hi``) characteristic and is the complement
	of the irrelevance-focused activation.

	Examples
	--------
	>>> import numpy as np
	>>> from machinegnostics.magnet import Dense, Relevance, Sequential
	>>> model = Sequential([Dense(2, 2), Relevance()])
	>>> model(np.array([[0.1, 0.2]])).shape
	(1, 2)
	"""

	def __init__(self, S: float | str = 1, name=None, verbose: bool = False):
		"""Create a relevance activation."""
		super().__init__(name, verbose=verbose)
		self.S = S

	def forward(self, x, training=True):
		"""Return the relevance characteristic for the supplied tensor."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		info = compute_characteristics(x.data, scale=self.S)
		self.S_local = info["S_local"]
		self.relevance = (np.asarray(info["hi"], dtype=np.float64))
		prime = (2.0 / self.S_local) * (1 - self.relevance ** 2)
		return _gnostic_activation_tensor(x, self.relevance, prime)

class Square(Activation):
	"""Square activation function.

	This activation returns the square of the input tensor and is useful
	when the model should emphasize the squared values.

	Examples
	--------
	>>> import numpy as np
	>>> from machinegnostics.magnet import Dense, Square, Sequential
	>>> model = Sequential([Dense(2, 2), Square()])
	>>> model(np.array([[0.1, 0.2]])).shape
	(1, 2)
	"""

	def forward(self, x, training=True):
		"""Return the square of the input tensor."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		data = np.square(x.data)
		prime = 2 * x.data
		return _gnostic_activation_tensor(x, data, prime)


def fi(x, S: float | str = 1):
	"""Convenience function returning the gnostic fidelity characteristic."""
	return np.asarray(compute_characteristics(x, scale=S)["fi"], dtype=np.float64)


def fj(x, S: float | str = 1):
	"""Convenience function returning the gnostic infidelity characteristic."""
	return np.asarray(compute_characteristics(x, scale=S)["fj"], dtype=np.float64)


def hi(x, S: float | str = 1):
	"""Convenience function returning the gnostic irrelevance characteristic."""
	return np.asarray(compute_characteristics(x, scale=S)["hj"], dtype=np.float64)


def hj(x, S: float | str = 1):
	"""Convenience function returning the gnostic relevance characteristic."""
	return np.asarray(compute_characteristics(x, scale=S)["hj"], dtype=np.float64)


def get_activation(activation, verbose: bool = False):
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
			"relu": ReLU,
			"step": Step,
			"threshold": Step,
			"heaviside": Step,
			"leakyrelu": LeakyReLU,
			"elu": ELU,
			"sigmoid": Sigmoid,
			"softplus": Softplus,
			"tanh": Tanh,
			"swish": Swish,
			"softmax": Softmax,
			"fidelity": Fidelity,
			"fiactivation": FiActivation,
			"infidelity": Infidelity,
			"irrelevance": Irrelevance,
			"relevance": Relevance,
			"gnosticproba": GnosticProba,
			"entropy": Entropy,
		}
		key = activation.replace("_", "").replace("-", "").lower()
		try:
			return registry[key](verbose=verbose)
		except KeyError as exc:
			raise ValueError(f"Unknown activation: {activation}") from exc
	raise TypeError(f"Unsupported activation specification: {type(activation)!r}")


from .gn_activations import ActivationFunctions