"""Dedicated public activation module for elu.

Developer note
--------------
This module keeps the public MAGNET activation API one class per file.
"""

from __future__ import annotations

import numpy as np
import torch

from ._shared import Tensor, compute_characteristics, custom_tensor, _gnostic_activation_tensor, Layer
from .activation import Activation

class ELU(Activation):
	"""Exponential linear unit activation.

	ELU behaves like an identity for positive inputs and transitions smoothly to
	a negative exponential for values below zero.

	Examples
	--------
	>>> import numpy as np
	>>> from machinegnostics.magnet import Dense, ELU, Sequential
	>>> model = Sequential([Dense(2, 2), ELU(alpha=1.0)])
	>>> model(np.array([[1.0, -1.0]])).shape
	(1, 2)
	"""

	def __init__(self, alpha: float = 1.0, name=None, verbose: bool = False):
		"""Create an ELU activation."""
		super().__init__(name, verbose=verbose)
		self.alpha = float(alpha)

	def forward(self, x, training=True):
		"""Return the elementwise ELU transform."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		positive = x.data > 0
		data = np.where(positive, x.data, self.alpha * (np.expm1(x.data)))
		prime = np.where(positive, 1.0, data + self.alpha)
		return _gnostic_activation_tensor(x, data, prime)
