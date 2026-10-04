"""Dedicated initializer module for initializer.

Developer note
--------------
This module keeps the MAGNET initializer API one class per file.
"""

from __future__ import annotations

import numpy as np

from ..core.tensor import Tensor
from .base import Initializer, _compute_fans


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
