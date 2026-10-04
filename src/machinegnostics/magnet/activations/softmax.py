"""Dedicated public activation module for softmax.

Developer note
--------------
This module keeps the public MAGNET activation API one class per file.
"""

from __future__ import annotations

import numpy as np
import torch

from ._shared import Tensor, compute_characteristics, custom_tensor, _gnostic_activation_tensor, Layer
from .activation import Activation

class Softmax(Activation):
	"""Stable softmax activation over the last axis.

	Softmax converts logits into probabilities that sum to 1 along the final
	dimension.

	Examples
	--------
	>>> import numpy as np
	>>> from machinegnostics.magnet import Dense, Sequential, Softmax
	>>> model = Sequential([Dense(3, 3), Softmax()])
	>>> model(np.array([[1.0, 2.0, 3.0]])).shape
	(1, 3)
	"""

	def forward(self, x, training=True):
		"""Convert logits to probabilities along the final dimension."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		shifted = x - Tensor(np.max(x.data, axis=-1, keepdims=True))
		exp = shifted.exp()
		return exp / exp.sum(axis=-1, keepdims=True)
