"""Dedicated initializer module for ones.

Developer note
--------------
This module keeps the MAGNET initializer API one class per file.
"""

from __future__ import annotations

import numpy as np

from ..core.tensor import Tensor
from .base import Initializer, _compute_fans


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
