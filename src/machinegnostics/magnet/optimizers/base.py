"""Optimizer base class and registry for MAGNET.

Developer note
-------------
Author: Nirmal Parmar

Examples
--------
>>> from machinegnostics.magnet.optimizers import get_optimizer
>>> get_optimizer("adam")
Adam()
"""

from __future__ import annotations

from typing import Iterable, Union

import numpy as np

from ..tensor import Tensor

OptimizerLike = Union[str, "Optimizer"]


class Optimizer:
	"""Base optimizer that updates tensor parameters in-place."""

	def __init__(self, learning_rate: float = 0.001, lr: float | None = None):
		"""Store the optimizer learning rate."""
		self.learning_rate = learning_rate if lr is None else lr

	def step(self, params: Iterable[Tensor]) -> None:
		"""Apply one optimization step to a list of parameters."""
		raise NotImplementedError

	def zero_grad(self, params: Iterable[Tensor]) -> None:
		"""Reset gradients on every parameter passed to the optimizer."""
		for param in params:
			if param.grad is not None:
				param.grad = np.zeros_like(param.data)


def get_optimizer(optimizer: OptimizerLike | None) -> Optimizer:
	"""Resolve a string or instance into an optimizer object."""

	from .adam import Adam
	from .sgd import SGD

	if optimizer is None:
		return Adam()
	if isinstance(optimizer, Optimizer):
		return optimizer
	name = optimizer.lower()
	registry = {
		"sgd": SGD(),
		"adam": Adam(),
	}
	try:
		return registry[name]
	except KeyError as exc:
		raise ValueError(f"Unknown optimizer: {optimizer}") from exc
