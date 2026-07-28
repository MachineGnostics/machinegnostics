"""Parameter initializers for magnet."""

from __future__ import annotations

import numpy as np

from .tensor import Tensor


class Initializer:
	def __call__(self, shape):
		raise NotImplementedError


class Zeros(Initializer):
	def __call__(self, shape):
		return Tensor(np.zeros(shape, dtype=np.float64))


class Ones(Initializer):
	def __call__(self, shape):
		return Tensor(np.ones(shape, dtype=np.float64))


class RandomNormal(Initializer):
	def __init__(self, mean: float = 0.0, stddev: float = 0.01, seed=None):
		self.mean = mean
		self.stddev = stddev
		self.rng = np.random.default_rng(seed)

	def __call__(self, shape):
		return Tensor(self.rng.normal(self.mean, self.stddev, size=shape))


class XavierUniform(Initializer):
	def __init__(self, seed=None):
		self.rng = np.random.default_rng(seed)

	def __call__(self, shape):
		fan_in, fan_out = _compute_fans(shape)
		limit = np.sqrt(6.0 / max(fan_in + fan_out, 1))
		return Tensor(self.rng.uniform(-limit, limit, size=shape))


class HeNormal(Initializer):
	def __init__(self, seed=None):
		self.rng = np.random.default_rng(seed)

	def __call__(self, shape):
		fan_in, _ = _compute_fans(shape)
		stddev = np.sqrt(2.0 / max(fan_in, 1))
		return Tensor(self.rng.normal(0.0, stddev, size=shape))


def _compute_fans(shape):
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
