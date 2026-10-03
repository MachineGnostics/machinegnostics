"""Optimizer base class and registry for MAGNET."""

from __future__ import annotations

import logging
from typing import Iterable, Union

import torch

from ..core.tensor import Tensor
from ..utils.logging import get_logger

OptimizerLike = Union[str, "Optimizer"]


class Optimizer:
    """Base class for MAGNET optimizers with adaptive special-parameter scaling."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        lr: float | None = None,
        gradient_scale_factor: float = 1.0,
        verbose: bool = False,
    ):
        self.learning_rate = learning_rate if lr is None else lr
        self.gradient_scale_factor = gradient_scale_factor
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if verbose else logging.WARNING)
        self.logger.debug("Optimizer initialized.")

    def step(self, params: Iterable[Tensor]) -> None:
        raise NotImplementedError

    def zero_grad(self, params: Iterable[Tensor]) -> None:
        for param in params:
            param.zero_grad()

    def get_param_name(self, param: Tensor) -> str:
        return (getattr(param, "name", None) or "").lower()

    def is_special_parameter(self, param_name: str) -> bool:
        return param_name == "s" or param_name == "z0" or param_name.endswith("_s") or param_name.endswith("_z0")

    def get_adaptive_lr(self, param_name: str, gradient: torch.Tensor) -> float:
        if self.is_special_parameter(param_name):
            grad_mag = float(torch.mean(torch.abs(gradient)).item()) + 1e-8
            return self.learning_rate / (1.0 + grad_mag * self.gradient_scale_factor)
        return self.learning_rate


def get_optimizer(optimizer: OptimizerLike | None) -> Optimizer:
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
