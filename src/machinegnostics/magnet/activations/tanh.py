"""Dedicated public activation module for tanh.

Developer note
--------------
This module keeps the public MAGNET activation API one class per file.
"""

from __future__ import annotations

import numpy as np
import torch

from ._shared import Tensor, compute_characteristics, custom_tensor, _gnostic_activation_tensor, Layer
from .activation import Activation

class Tanh(Activation):
	"""Hyperbolic tangent activation.

	Tanh squashes inputs into $(-1, 1)$ and is often used in hidden layers.

	Examples
	--------
	>>> import numpy as np
	>>> from machinegnostics.magnet import Dense, Sequential, Tanh
	>>> model = Sequential([Dense(3, 2), Tanh()])
	>>> model(np.array([[1.0, 0.0, -1.0]])).shape
	(1, 2)
	"""

	def forward(self, x, training=True):
		"""Return the elementwise hyperbolic tangent of the input."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		return x.tanh()
