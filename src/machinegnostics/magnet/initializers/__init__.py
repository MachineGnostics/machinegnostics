"""Parameter initializers for MAGNET (Machine Gnostics Neural Networks).

Developer note
-------------
Author: Nirmal Parmar

The public API is intentionally simple:

- instantiate an initializer class directly when you want full control;
- pass a string to ``get_initializer`` when you want a shortcut;
- call the initializer with a shape tuple to receive a ``Tensor``.

The returned tensors are NumPy-backed MAGNET tensors, so they can be assigned
directly into layer parameter dictionaries and participate in autograd.

Examples
--------
>>> from machinegnostics.magnet.initializers import XavierUniform
>>> w = XavierUniform(seed=4)((2, 3))
>>> w.shape
(2, 3)
"""

from __future__ import annotations

import numpy as np

from ..core.tensor import Tensor


class Initializer:
	"""Base class for all MAGNET parameter initializers.

	An initializer is a small callable object that receives a target shape and
	returns a MAGNET ``Tensor`` containing the initial parameter values.

	Subclasses only need to implement ``__call__``.

	Examples
	--------
	>>> from machinegnostics.magnet import Initializer
	>>> isinstance(Initializer(), Initializer)
	True
	"""

	def __call__(self, shape):
		"""Generate an initialized tensor for ``shape``.

		Parameters
		----------
		shape:
			Target tensor shape, typically a tuple of integers.

		Returns
		-------
		Tensor
			Initialized parameter tensor.
		"""
		raise NotImplementedError


class Zeros(Initializer):
	"""Return all-zero tensors, typically used for biases.

	``Zeros`` is the default bias initializer in many MAGNET layers because
	it starts the affine offset at a neutral value.

	Examples
	--------
	>>> from machinegnostics.magnet import Zeros
	>>> Zeros()((2, 3)).data
	array([[0., 0., 0.],
		   [0., 0., 0.]])
	"""

	def __call__(self, shape):
		"""Return a zero-filled tensor with the requested shape."""
		return Tensor(np.zeros(shape, dtype=np.float64))


class Ones(Initializer):
	"""Return all-one tensors.

	``Ones`` is useful when you want a deterministic starting point or a
	simple placeholder initializer during debugging.

	Examples
	--------
	>>> from machinegnostics.magnet import Ones
	>>> Ones()((2, 2)).data
	array([[1., 1.],
		   [1., 1.]])
	"""

	def __call__(self, shape):
		"""Return a one-filled tensor with the requested shape."""
		return Tensor(np.ones(shape, dtype=np.float64))


class RandomNormal(Initializer):
	"""Sample parameters from a Gaussian distribution.

	``RandomNormal`` draws each entry independently from ``N(mean, stddev)``.
	It is useful when you want a small random starting point that can be tuned
	by supplying a seed.

	Examples
	--------
	>>> init = RandomNormal(seed=7)
	>>> init((2, 2)).shape
	(2, 2)
	"""
	def __init__(self, mean: float = 0.0, stddev: float = 0.01, seed=None):
		"""Create a normal initializer.

		Parameters
		----------
		mean:
			Mean of the normal distribution.
		stddev:
			Standard deviation of the normal distribution.
		seed:
			Optional random seed for reproducibility.

		Examples
		--------
		>>> from machinegnostics.magnet import RandomNormal
		>>> init = RandomNormal(mean=0.0, stddev=0.1, seed=7)
		>>> init((2, 2)).shape
		(2, 2)
		"""
		self.mean = mean
		self.stddev = stddev
		self.rng = np.random.default_rng(seed)

	def __call__(self, shape):
		"""Draw a tensor of the requested shape from the configured normal distribution.

		Parameters
		----------
		shape:
			Target tensor shape.

		Returns
		-------
		Tensor
			Tensor whose entries are sampled from the configured Gaussian.
		"""
		return Tensor(self.rng.normal(self.mean, self.stddev, size=shape))


class XavierUniform(Initializer):
	"""Glorot/Xavier uniform initialization.

	Use this for sigmoid, tanh, and many small feed-forward MAGNET models.
	The distribution bounds are computed from the fan-in and fan-out of the
	target shape.

	Examples
	--------
	>>> from machinegnostics.magnet import XavierUniform
	>>> XavierUniform(seed=4)((2, 3)).shape
	(2, 3)
	"""
	def __init__(self, seed=None):
		"""Create a Xavier uniform initializer.

		Parameters
		----------
		seed:
			Optional random seed for reproducibility.
		"""
		self.rng = np.random.default_rng(seed)

	def __call__(self, shape):
		"""Generate a Xavier-initialized tensor for the requested shape.

		Parameters
		----------
		shape:
			Weight tensor shape.

		Returns
		-------
		Tensor
			Tensor sampled from the Xavier uniform range.
		"""
		fan_in, fan_out = _compute_fans(shape)
		limit = np.sqrt(6.0 / max(fan_in + fan_out, 1))
		return Tensor(self.rng.uniform(-limit, limit, size=shape))


class HeNormal(Initializer):
	"""He normal initialization for ReLU-style networks.

	``HeNormal`` is a good default for layers followed by ReLU-family
	activations because it preserves activation scale more effectively than a
	plain zero-centered Gaussian for deep rectified networks.

	Examples
	--------
	>>> from machinegnostics.magnet import HeNormal
	>>> HeNormal(seed=2)((4, 3)).shape
	(4, 3)
	"""
	def __init__(self, seed=None):
		"""Create a He normal initializer.

		Parameters
		----------
		seed:
			Optional random seed for reproducibility.
		"""
		self.rng = np.random.default_rng(seed)

	def __call__(self, shape):
		"""Generate a He-initialized tensor for the requested shape.

		Parameters
		----------
		shape:
			Weight tensor shape.

		Returns
		-------
		Tensor
			Tensor sampled from the He normal distribution.
		"""
		fan_in, _ = _compute_fans(shape)
		stddev = np.sqrt(2.0 / max(fan_in, 1))
		return Tensor(self.rng.normal(0.0, stddev, size=shape))


def _compute_fans(shape):
	"""Compute fan-in and fan-out for a weight tensor shape.

	The helper supports the common dense ``(in, out)`` layout, convolutional
	``(out_ch, in_ch, kh, kw)`` kernels, and a fallback for any other shape by
	using the total element count.
	"""
	if len(shape) == 2:
		fan_in, fan_out = shape
	elif len(shape) == 4:
		out_ch, in_ch, kh, kw = shape
		receptive_field = kh * kw
		fan_in = in_ch * receptive_field
		fan_out = out_ch * receptive_field
	else:
		fan_in = fan_out = int(np.prod(shape))
	return fan_in, fan_out


def get_initializer(initializer):
	"""Resolve an initializer specification into an initializer instance.

	Parameters
	----------
	initializer:
		``None`` for the default Xavier initializer, an initializer instance,
		or a string such as ``"zeros"``, ``"ones"``, ``"randomnormal"``,
		``"xavieruniform"``, or ``"henormal"``.

	Returns
	-------
	Initializer
		Resolved initializer object.

	Examples
	--------
	>>> from machinegnostics.magnet import get_initializer
	>>> get_initializer("xavieruniform")
	XavierUniform()
	"""
	if initializer is None:
		return XavierUniform()
	if isinstance(initializer, Initializer):
		return initializer
	if isinstance(initializer, str):
		registry = {
			"zeros": Zeros(),
			"ones": Ones(),
			"randomnormal": RandomNormal(),
			"xavieruniform": XavierUniform(),
			"henormal": HeNormal(),
		}
		key = initializer.replace("_", "").replace("-", "").lower()
		try:
			return registry[key]
		except KeyError as exc:
			raise ValueError(f"Unknown initializer: {initializer}") from exc
	raise TypeError(f"Unsupported initializer specification: {type(initializer)!r}")
