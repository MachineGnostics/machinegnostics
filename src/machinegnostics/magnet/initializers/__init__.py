"""Parameter initializers for MAGNET (Machine Gnostics Neural Networks).

Developer note
-------------
Author: Nirmal Parmar

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
	"""Base class for all MAGNET parameter initializers."""

	def __call__(self, shape):
		"""Generate an initialized tensor for ``shape``."""
		raise NotImplementedError


class Zeros(Initializer):
	"""Return all-zero tensors, typically used for biases."""

	def __call__(self, shape):
		return Tensor(np.zeros(shape, dtype=np.float64))


class Ones(Initializer):
	"""Return all-one tensors."""

	def __call__(self, shape):
		return Tensor(np.ones(shape, dtype=np.float64))


class RandomNormal(Initializer):
	"""Sample parameters from a Gaussian distribution.

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
		"""
		self.mean = mean
		self.stddev = stddev
		self.rng = np.random.default_rng(seed)

	def __call__(self, shape):
		"""Draw a tensor of the requested shape from the configured normal distribution."""
		return Tensor(self.rng.normal(self.mean, self.stddev, size=shape))


class XavierUniform(Initializer):
	"""Glorot/Xavier uniform initialization.

	Use this for sigmoid, tanh, and many small feed-forward MAGNET models.
	"""
	def __init__(self, seed=None):
		"""Create a Xavier uniform initializer."""
		self.rng = np.random.default_rng(seed)

	def __call__(self, shape):
		"""Generate a Xavier-initialized tensor for the requested shape."""
		fan_in, fan_out = _compute_fans(shape)
		limit = np.sqrt(6.0 / max(fan_in + fan_out, 1))
		return Tensor(self.rng.uniform(-limit, limit, size=shape))


class HeNormal(Initializer):
	"""He normal initialization for ReLU-style networks."""
	def __init__(self, seed=None):
		"""Create a He normal initializer."""
		self.rng = np.random.default_rng(seed)

	def __call__(self, shape):
		"""Generate a He-initialized tensor for the requested shape."""
		fan_in, _ = _compute_fans(shape)
		stddev = np.sqrt(2.0 / max(fan_in, 1))
		return Tensor(self.rng.normal(0.0, stddev, size=shape))


def _compute_fans(shape):
	"""Compute fan-in and fan-out for a weight tensor shape."""
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
	"""Resolve an initializer specification into an initializer instance."""
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
