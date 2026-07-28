"""Stochastic gradient descent optimizer."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ..tensor import Tensor
from .base import Optimizer


class SGD(Optimizer):
	def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0, lr: float | None = None):
		super().__init__(learning_rate=learning_rate, lr=lr)
		self.momentum = momentum
		self._velocity = {}

	def step(self, params: Iterable[Tensor]) -> None:
		for param in params:
			if param.grad is None:
				continue
			key = id(param)
			velocity = self._velocity.get(key, np.zeros_like(param.data))
			velocity = self.momentum * velocity - self.learning_rate * param.grad
			self._velocity[key] = velocity
			param.data = param.data + velocity
