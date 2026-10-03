"""Adam optimizer for MAGNET."""

from __future__ import annotations

from typing import Iterable

import torch

from ..core.tensor import Tensor
from .base import Optimizer


class Adam(Optimizer):
    """Adam optimizer with automatic scaling for MAGNET special parameters."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
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
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self._m = {}
        self._v = {}
        self._t = 0

    def step(self, params: Iterable[Tensor]) -> None:
        self._t += 1
        for param in params:
            if param._tensor.grad is None:
                continue
            key = id(param)
            m = self._m.get(key)
            v = self._v.get(key)
            if m is None:
                m = torch.zeros_like(param._tensor)
            if v is None:
                v = torch.zeros_like(param._tensor)

            grad = param._tensor.grad
            m = self.beta1 * m + (1.0 - self.beta1) * grad
            v = self.beta2 * v + (1.0 - self.beta2) * (grad ** 2)

            m_hat = m / (1.0 - self.beta1 ** self._t)
            v_hat = v / (1.0 - self.beta2 ** self._t)
            lr = self.get_adaptive_lr(self.get_param_name(param), grad)

            with torch.no_grad():
                param._tensor.add_(-lr * m_hat / (torch.sqrt(v_hat) + self.epsilon))

            self._m[key] = m.detach()
            self._v[key] = v.detach()
