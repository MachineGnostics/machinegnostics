"""Dedicated initializer module for he_normal.

Developer note
--------------
This module keeps the MAGNET initializer API one class per file.
"""

from __future__ import annotations

import numpy as np

from ..core.tensor import Tensor
from .base import Initializer, _compute_fans


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
