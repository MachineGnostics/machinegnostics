"""Stochastic gradient descent optimizer for MAGNET.

Developer note
-------------
Author: Nirmal Parmar

Examples
--------
>>> from machinegnostics.magnet import SGD
>>> SGD(lr=0.01)
SGD(...)
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ..tensor import Tensor
from .base import Optimizer


class SGD(Optimizer):
	"""Stochastic gradient descent with optional momentum.

	SGD is the simplest optimizer in the MAGNET stack. It updates each
	parameter directly using the current gradient and, if requested, a momentum
	buffer that smooths the update direction.

	Examples
	--------
	>>> from machinegnostics.magnet import SGD
	>>> SGD(lr=0.01, momentum=0.9)
	SGD(...)
	"""

	def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0, lr: float | None = None, verbose: bool = False):
		"""Create an SGD optimizer.

		Parameters
		----------
		learning_rate:
			Step size used for updates.
		momentum:
			Momentum coefficient.
		lr:
			Alias for ``learning_rate``.
		verbose:
			Enable debug logging for the optimizer instance.
		"""
		super().__init__(learning_rate=learning_rate, lr=lr, verbose=verbose)
		self.momentum = momentum
		self._velocity = {}

	def step(self, params: Iterable[Tensor]) -> None:
		"""Update each parameter tensor using SGD with momentum."""
		for param in params:
			if param.grad is None:
				continue
			key = id(param)
			velocity = self._velocity.get(key, np.zeros_like(param.data))
			velocity = self.momentum * velocity - self.learning_rate * param.grad
			self._velocity[key] = velocity
			param.data = param.data + velocity
