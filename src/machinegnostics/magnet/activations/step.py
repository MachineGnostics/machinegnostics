"""Dedicated public activation module for step.

Developer note
--------------
This module keeps the public MAGNET activation API one class per file.
"""

from __future__ import annotations

import numpy as np
import torch

from ._shared import Tensor, compute_characteristics, custom_tensor, _gnostic_activation_tensor, Layer
from .activation import Activation

class Step(Activation):
	"""Hard step activation that maps values to 0 or 1.

	Step is useful for thresholding and discrete gating. It is not smooth, so
	it should be used when a binary output is more important than gradient-rich
	training dynamics.

	Examples
	--------
	>>> import numpy as np
	>>> from machinegnostics.magnet import Dense, Sequential, Step
	>>> model = Sequential([Dense(3, 2), Step()])
	>>> model(np.array([[0.0, 1.0, -1.0]])).shape
	(1, 2)
	"""

	def __init__(self, threshold: float = 0.0, name=None, verbose: bool = False):
		"""Create a step activation."""
		super().__init__(name, verbose=verbose)
		self.threshold = float(threshold)

	def forward(self, x, training=True):
		"""Return 1 where the input exceeds the threshold, else 0."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		data = (x.data > self.threshold).astype(np.float64)
		prime = np.zeros_like(x.data, dtype=np.float64)
		return _gnostic_activation_tensor(x, data, prime)
