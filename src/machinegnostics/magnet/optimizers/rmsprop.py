"""RMSprop optimizer for MAGNET.

Developer note
-------------
Author: Nirmal Parmar

Examples
--------
>>> from machinegnostics.magnet import RMSprop
>>> RMSprop(lr=0.001)
RMSprop(...)
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from ..core.tensor import Tensor
from .base import Optimizer


class RMSprop(Optimizer):
	"""RMSprop optimizer with an exponential moving average of squared gradients.

	RMSprop keeps a running average of gradient magnitudes so the effective
	step size stays stable across parameters.

	Examples
	--------
	>>> from machinegnostics.magnet import RMSprop
	>>> RMSprop(lr=0.001)
	RMSprop(...)
	"""

	def __init__(self, learning_rate: float = 0.001, rho: float = 0.9, epsilon: float = 1e-8, lr: float | None = None, verbose: bool = False):
		"""Create an RMSprop optimizer.

		Parameters
		----------
		learning_rate:
			Step size used for updates.
		rho:
			Decay factor for the moving average of squared gradients.
		epsilon:
			Small constant that prevents division by zero.
		lr:
			Alias for ``learning_rate``.
		verbose:
			Enable debug logging for the optimizer instance.
		"""
		super().__init__(learning_rate=learning_rate, lr=lr, verbose=verbose)
		self.rho = rho
		self.epsilon = epsilon
		self._cache = {}

	def step(self, params: Iterable[Tensor]) -> None:
		"""Update each parameter tensor using RMSprop."""
		for param in params:
			if param._tensor.grad is None:
				continue
			key = id(param)
			cache = self._cache.get(key)
			if cache is None:
				cache = torch.zeros_like(param._tensor)
			grad = param._tensor.grad
			cache = self.rho * cache + (1.0 - self.rho) * (grad ** 2)
			with torch.no_grad():
				param._tensor.add_(-self.learning_rate * grad / (torch.sqrt(cache) + self.epsilon))
			self._cache[key] = cache.detach()