"""Dedicated public activation module for square.

Developer note
--------------
This module keeps the public MAGNET activation API one class per file.
"""

from __future__ import annotations

import numpy as np
import torch

from ._shared import Tensor, compute_characteristics, custom_tensor, _gnostic_activation_tensor, Layer
from .activation import Activation

class Square(Activation):
	"""Square activation function.

	This activation returns the square of the input tensor and is useful
	when the model should emphasize the squared values.

	Examples
	--------
	>>> import numpy as np
	>>> from machinegnostics.magnet import Dense, Square, Sequential
	>>> model = Sequential([Dense(2, 2), Square()])
	>>> model(np.array([[0.1, 0.2]])).shape
	(1, 2)
	"""

	def forward(self, x, training=True):
		"""Return the square of the input tensor."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		data = np.square(x.data)
		prime = 2 * x.data
		return _gnostic_activation_tensor(x, data, prime)
