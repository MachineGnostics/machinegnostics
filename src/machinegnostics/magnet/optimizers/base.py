"""Optimizer base class and registry."""

from __future__ import annotations

from typing import Iterable, Union

import numpy as np

from ..tensor import Tensor

OptimizerLike = Union[str, "Optimizer"]


class Optimizer:
	"""Base optimizer that updates tensors in-place."""

	def __init__(self, learning_rate: float = 0.001, lr: float | None = None):
		self.learning_rate = learning_rate if lr is None else lr

	def step(self, params: Iterable[Tensor]) -> None:
		raise NotImplementedError

	def zero_grad(self, params: Iterable[Tensor]) -> None:
		for param in params:
			if param.grad is not None:
				param.grad = np.zeros_like(param.data)


def get_optimizer(optimizer: OptimizerLike | None) -> Optimizer:
	"""Resolve an optimizer specification into an optimizer instance."""

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
