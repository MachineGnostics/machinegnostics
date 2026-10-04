"""Dedicated public activation module for relu.

Developer note
--------------
This module keeps the public MAGNET activation API one class per file.
"""

from __future__ import annotations

import numpy as np
import torch

from ._shared import Tensor, compute_characteristics, custom_tensor, _gnostic_activation_tensor, Layer
from .activation import Activation

class ReLU(Activation):
	"""Rectified linear unit activation.

	ReLU returns zero for negative inputs and preserves positive values.

	Examples
	--------
	>>> import numpy as np
	>>> from machinegnostics.magnet import Dense, ReLU, Sequential
	>>> model = Sequential([Dense(2, 2), ReLU()])
	>>> model(np.array([[1.0, -2.0]])).shape
	(1, 2)
	"""

	def forward(self, x, training=True):
		"""Return ``max(0, x)`` elementwise."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		return x.relu()
