"""Stochastic gradient descent optimizer for MAGNET."""

from __future__ import annotations

from typing import Iterable

import torch

from ..core.tensor import Tensor
from .base import Optimizer


class SGD(Optimizer):
    """SGD with automatic gradient-based scaling for ``S`` and ``z0``."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        lr: float | None = None,
        gradient_scale_factor: float = 1.0,
        verbose: bool = False,
    ):
        super().__init__(
            learning_rate=learning_rate,
            lr=lr,
            gradient_scale_factor=gradient_scale_factor,
            verbose=verbose,
        )
        self.momentum = momentum
        self._velocity = {}

    def step(self, params: Iterable[Tensor]) -> None:
        for param in params:
            if param._tensor.grad is None:
                continue
            key = id(param)
            velocity = self._velocity.get(key)
            if velocity is None:
                velocity = torch.zeros_like(param._tensor)
            grad = param._tensor.grad
            lr = self.get_adaptive_lr(self.get_param_name(param), grad)
            with torch.no_grad():
                velocity = self.momentum * velocity - lr * grad
                param._tensor.add_(velocity)
            self._velocity[key] = velocity.detach()
