"""Optimizer base class and registry for MAGNET.

Developer note
-------------
Author: Nirmal Parmar

Examples
--------
>>> from machinegnostics.magnet import get_optimizer
>>> get_optimizer("adam")
Adam()
"""

from __future__ import annotations

import logging
from typing import Iterable, Union

import numpy as np

from ..core.tensor import Tensor
from ..utils.logging import get_logger

OptimizerLike = Union[str, "Optimizer"]


class Optimizer:
	"""Base class for MAGNET optimizers.

	The base class owns the shared learning-rate bookkeeping, a lightweight
	logger, and the ``zero_grad`` helper used by all concrete optimizers.
	Subclasses only need to implement ``step``.

	Examples
	--------
	>>> from machinegnostics.magnet.optimizers import Optimizer
	>>> isinstance(Optimizer(learning_rate=0.01), Optimizer)
	True
	"""

	def __init__(self, learning_rate: float = 0.001, lr: float | None = None, verbose: bool = False):
		"""Store the optimizer learning rate and configure logging."""
		self.learning_rate = learning_rate if lr is None else lr
		self.logger = get_logger(self.__class__.__name__, logging.DEBUG if verbose else logging.WARNING)
		self.logger.debug("Optimizer initialized.")

	def step(self, params: Iterable[Tensor]) -> None:
		"""Apply one optimization step to a list of parameters."""
		raise NotImplementedError

	def zero_grad(self, params: Iterable[Tensor]) -> None:
		"""Reset gradients on every parameter passed to the optimizer."""
		for param in params:
			param.zero_grad()


def get_optimizer(optimizer: OptimizerLike | None) -> Optimizer:
	"""Resolve a string or instance into an optimizer object."""

	from .adam import Adam
	from .adagrad import Adagrad
	from .sgd import SGD
	from .rmsprop import RMSprop

	if optimizer is None:
		return Adam()
	if isinstance(optimizer, Optimizer):
		return optimizer
	name = optimizer.lower()
	registry = {
		"adagrad": Adagrad(),
		"sgd": SGD(),
		"adam": Adam(),
		"rmsprop": RMSprop(),
	}
	try:
		return registry[name]
	except KeyError as exc:
		raise ValueError(f"Unknown optimizer: {optimizer}") from exc
