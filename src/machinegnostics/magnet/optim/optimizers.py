"""Optimizers for Magnet ANN training.

Author: Nirmal Parmar

Notes:
- Includes SGD and Adam with standard hyperparameters.
- Optimizers consume iterable `Parameter` tensors.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

from ..layers import Parameter


class Optimizer:
    """Base optimizer interface."""

    def __init__(self, params: Iterable[Parameter], lr: float = 1e-3):
        self.params: List[Parameter] = list(params)
        self.lr = float(lr)

    def zero_grad(self) -> None:
        for param in self.params:
            param.zero_grad()

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent with optional momentum and weight decay."""

    def __init__(self, params: Iterable[Parameter], lr=1e-2, momentum=0.0, weight_decay=0.0):
        super().__init__(params=params, lr=lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self._velocity: Dict[int, np.ndarray] = {}

    def step(self) -> None:
        for param in self.params:
            if param.grad is None:
                continue
            grad = param.grad
            if self.weight_decay > 0.0:
                grad = grad + self.weight_decay * param.data

            if self.momentum > 0.0:
                pid = id(param)
                if pid not in self._velocity:
                    self._velocity[pid] = np.zeros_like(param.data)
                self._velocity[pid] = self.momentum * self._velocity[pid] + grad
                grad = self._velocity[pid]

            param.data -= self.lr * grad


class Adam(Optimizer):
    """Adam optimizer with optional decoupled weight decay off by default."""

    def __init__(self, params: Iterable[Parameter], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params=params, lr=lr)
        self.beta1 = float(betas[0])
        self.beta2 = float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.t = 0
        self._m: Dict[int, np.ndarray] = {}
        self._v: Dict[int, np.ndarray] = {}

    def step(self) -> None:
        self.t += 1
        for param in self.params:
            if param.grad is None:
                continue
            grad = param.grad
            if self.weight_decay > 0.0:
                grad = grad + self.weight_decay * param.data

            pid = id(param)
            if pid not in self._m:
                self._m[pid] = np.zeros_like(param.data)
                self._v[pid] = np.zeros_like(param.data)

            self._m[pid] = self.beta1 * self._m[pid] + (1.0 - self.beta1) * grad
            self._v[pid] = self.beta2 * self._v[pid] + (1.0 - self.beta2) * (grad * grad)

            m_hat = self._m[pid] / (1.0 - self.beta1**self.t)
            v_hat = self._v[pid] / (1.0 - self.beta2**self.t)

            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
