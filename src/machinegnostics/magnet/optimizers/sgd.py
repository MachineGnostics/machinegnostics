"""Stochastic gradient descent optimizer for MAGNET.

Developer note
-------------
Author: Nirmal Parmar

Examples
--------
>>> from machinegnostics.magnet.optimizers import SGD
>>> SGD(lr=0.01)
SGD(...)
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ..tensor import Tensor
from .base import Optimizer


class SGD(Optimizer):
	"""Classical SGD with optional momentum."""

	def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0, lr: float | None = None):
		"""Create an SGD optimizer.

		Parameters
		----------
		learning_rate:
			Step size used for updates.
		momentum:
			Momentum coefficient.
		lr:
			Alias for ``learning_rate``.
		"""
		super().__init__(learning_rate=learning_rate, lr=lr)
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
