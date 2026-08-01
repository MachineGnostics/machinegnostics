"""Adagrad optimizer for MAGNET.

Developer note
-------------
Author: Nirmal Parmar

Examples
--------
>>> from machinegnostics.magnet import Adagrad
>>> Adagrad(lr=0.01)
Adagrad(...)
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from ..core.tensor import Tensor
from .base import Optimizer


class Adagrad(Optimizer):
	"""Adagrad optimizer with per-parameter accumulated squared gradients.

	Adagrad adapts the learning rate for each parameter using the history of
	its squared gradients, which can be helpful when features have very
	different scales.

	Examples
	--------
	>>> from machinegnostics.magnet import Adagrad
	>>> Adagrad(lr=0.01)
	Adagrad(...)
	"""

	def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-8, lr: float | None = None, verbose: bool = False):
		"""Create an Adagrad optimizer.

		Parameters
		----------
		learning_rate:
			Step size used for updates.
		epsilon:
			Small constant that prevents division by zero.
		lr:
			Alias for ``learning_rate``.
		verbose:
			Enable debug logging for the optimizer instance.
		"""
		super().__init__(learning_rate=learning_rate, lr=lr, verbose=verbose)
		self.epsilon = epsilon
		self._cache = {}

	def step(self, params: Iterable[Tensor]) -> None:
		"""Update each parameter tensor using the Adagrad rule."""
		for param in params:
			if param._tensor.grad is None:
				continue
			key = id(param)
			cache = self._cache.get(key)
			if cache is None:
				cache = torch.zeros_like(param._tensor)
			grad = param._tensor.grad
			cache = cache + grad ** 2
			with torch.no_grad():
				param._tensor.add_(-self.learning_rate * grad / (torch.sqrt(cache) + self.epsilon))
			self._cache[key] = cache.detach()