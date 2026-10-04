"""Dedicated initializer module for zeros.

Developer note
--------------
This module keeps the MAGNET initializer API one class per file.
"""

from __future__ import annotations

import numpy as np

from ..core.tensor import Tensor
from .base import Initializer, _compute_fans


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
