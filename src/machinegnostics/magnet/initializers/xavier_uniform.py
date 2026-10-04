"""Dedicated initializer module for xavier_uniform.

Developer note
--------------
This module keeps the MAGNET initializer API one class per file.
"""

from __future__ import annotations

import numpy as np

from ..core.tensor import Tensor
from .base import Initializer, _compute_fans


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
