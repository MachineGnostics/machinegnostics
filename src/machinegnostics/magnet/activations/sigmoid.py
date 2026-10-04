"""Dedicated public activation module for sigmoid.

Developer note
--------------
This module keeps the public MAGNET activation API one class per file.
"""

from __future__ import annotations

import numpy as np
import torch

from ._shared import Tensor, compute_characteristics, custom_tensor, _gnostic_activation_tensor, Layer
from .activation import Activation

class Sigmoid(Activation):
	"""Logistic sigmoid activation.

	Sigmoid maps values into the open interval $(0, 1)$ and is commonly used
	for binary classification output layers.

	Examples
	--------
	>>> import numpy as np
	>>> from machinegnostics.magnet import Dense, Sequential, Sigmoid
	>>> model = Sequential([Dense(2, 1), Sigmoid()])
	>>> model(np.array([[0.0, 0.0]])).shape
	(1, 1)
	"""

	def forward(self, x, training=True):
		"""Return the elementwise sigmoid of the input."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		return x.sigmoid()
