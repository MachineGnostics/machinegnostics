"""Dedicated public activation module for swish.

Developer note
--------------
This module keeps the public MAGNET activation API one class per file.
"""

from __future__ import annotations

import numpy as np
import torch

from ._shared import Tensor, compute_characteristics, custom_tensor, _gnostic_activation_tensor, Layer
from .activation import Activation

class Swish(Activation):
	"""Swish activation, defined as ``x * sigmoid(x)``.

	Swish is a smooth, self-gated activation that often performs well as a
	drop-in alternative to ReLU in deep networks.

	Examples
	--------
	>>> import numpy as np
	>>> from machinegnostics.magnet import Dense, Sequential, Swish
	>>> model = Sequential([Dense(2, 2), Swish()])
	>>> model(np.array([[0.0, 1.0]])).shape
	(1, 2)
	"""

	def forward(self, x, training=True):
		"""Return the elementwise swish transform."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		sigmoid = 1.0 / (1.0 + np.exp(-np.clip(x.data, -500, 500)))
		data = x.data * sigmoid
		prime = sigmoid + x.data * sigmoid * (1.0 - sigmoid)
		return _gnostic_activation_tensor(x, data, prime)
