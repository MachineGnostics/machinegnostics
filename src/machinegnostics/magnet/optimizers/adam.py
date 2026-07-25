"""Adam optimizer."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ..tensor import Tensor
from .base import Optimizer


class Adam(Optimizer):
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate=learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self._m = {}
        self._v = {}
        self._t = 0

    def step(self, params: Iterable[Tensor]) -> None:
        self._t += 1
        for param in params:
            if param.grad is None:
                continue
            key = id(param)
            m = self._m.get(key, np.zeros_like(param.data))
            v = self._v.get(key, np.zeros_like(param.data))

            m = self.beta1 * m + (1.0 - self.beta1) * param.grad
            v = self.beta2 * v + (1.0 - self.beta2) * (param.grad ** 2)

            m_hat = m / (1.0 - self.beta1 ** self._t)
            v_hat = v / (1.0 - self.beta2 ** self._t)

            param.data = param.data - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

            self._m[key] = m
            self._v[key] = v
