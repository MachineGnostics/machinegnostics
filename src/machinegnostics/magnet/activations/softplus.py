"""Dedicated public activation module for softplus.

Developer note
--------------
This module keeps the public MAGNET activation API one class per file.
"""

from __future__ import annotations

import numpy as np
import torch

from ._shared import Tensor, compute_characteristics, custom_tensor, _gnostic_activation_tensor, Layer
from .activation import Activation

class Softplus(Activation):
	"""Softplus activation.

	Softplus is a smooth approximation of ReLU and is useful when a strictly
	positive, differentiable output is desired.

	Examples
	--------
	>>> import numpy as np
	>>> from machinegnostics.magnet import Dense, Sequential, Softplus
	>>> model = Sequential([Dense(2, 1), Softplus()])
	>>> model(np.array([[0.0, 0.0]])).shape
	(1, 1)
	"""

	def forward(self, x, training=True):
		"""Return a smooth approximation of ReLU."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		data = np.log1p(np.exp(-np.abs(x.data))) + np.maximum(x.data, 0)
		prime = 1.0 / (1.0 + np.exp(-np.clip(x.data, -500, 500)))
		return _gnostic_activation_tensor(x, data, prime)
