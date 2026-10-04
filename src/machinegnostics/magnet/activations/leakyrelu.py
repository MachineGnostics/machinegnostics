"""Dedicated public activation module for leakyrelu.

Developer note
--------------
This module keeps the public MAGNET activation API one class per file.
"""

from __future__ import annotations

import numpy as np
import torch

from ._shared import Tensor, compute_characteristics, custom_tensor, _gnostic_activation_tensor, Layer
from .activation import Activation

class LeakyReLU(Activation):
	"""Leaky rectified linear unit activation.

	LeakyReLU keeps a small slope for negative inputs so gradients can flow even
	when activations fall below zero.

	Examples
	--------
	>>> import numpy as np
	>>> from machinegnostics.magnet import Dense, LeakyReLU, Sequential
	>>> model = Sequential([Dense(2, 2), LeakyReLU(alpha=0.1)])
	>>> model(np.array([[-2.0, 3.0]])).shape
	(1, 2)
	"""

	def __init__(self, alpha: float = 0.01, name=None, verbose: bool = False):
		"""Create a leaky ReLU activation."""
		super().__init__(name, verbose=verbose)
		self.alpha = float(alpha)

	def forward(self, x, training=True):
		"""Return ``x`` for positive inputs and ``alpha * x`` otherwise."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		data = np.where(x.data > 0, x.data, self.alpha * x.data)
		prime = np.where(x.data > 0, 1.0, self.alpha)
		return _gnostic_activation_tensor(x, data, prime)
