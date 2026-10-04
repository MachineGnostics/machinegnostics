"""Dedicated initializer module for random_normal.

Developer note
--------------
This module keeps the MAGNET initializer API one class per file.
"""

from __future__ import annotations

import numpy as np

from ..core.tensor import Tensor
from .base import Initializer, _compute_fans


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
