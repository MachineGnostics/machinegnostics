"""Initializer base utilities for MAGNET.

Developer note
--------------
Author: Nirmal Parmar, OSS Machine Gnostics

This module defines the base initializer protocol and the fan-in/fan-out
shape analysis shared by the concrete initializer implementations.
"""

from __future__ import annotations

import numpy as np

from ..core.tensor import Tensor


class Initializer:
	"""Base class for all MAGNET parameter initializers."""

	def __call__(self, shape):
		"""Return initialized values for ``shape``."""
		raise NotImplementedError


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
