from abc import ABCMeta, abstractmethod

import numpy as np


class Optimizer(metaclass=ABCMeta):
    def __init__(self, lr: float = 0.01):
        self.lr = float(lr)

    @abstractmethod
    def step(self, params: dict, grads: dict, scale: float = 1.0):
        """Apply an update given params and grads dictionaries.

        - params: mapping 'layer:param' -> ndarray (weights)
        - grads: mapping 'layer:param' -> ndarray (gradients)
        - scale: optional global scaling factor (e.g., gnostic influence)
        """
        pass


class SGD(Optimizer):
    def __init__(self, lr: float = 0.01):
        super().__init__(lr=lr)

    def step(self, params: dict, grads: dict, scale: float = 1.0):
        for k, g in grads.items():
            if k in params:
                params[k][:] = params[k] - (self.lr * scale) * g


class Momentum(Optimizer):
    def __init__(self, lr: float = 0.001, momentum: float = 0.9):
        super().__init__(lr=lr)
        self.momentum = float(momentum)
        self._mv = {}

    def step(self, params: dict, grads: dict, scale: float = 1.0):
        for k, g in grads.items():
            if k not in params:
                continue
            v = self._mv.get(k)
            if v is None:
                v = np.zeros_like(params[k])
            dv = (self.lr * scale) * g
            v = self.momentum * v + dv
            params[k][:] = params[k] - v
            self._mv[k] = v


class AdaGrad(Optimizer):
    def __init__(self, lr: float = 0.01, eps: float = 1e-06):
        super().__init__(lr=lr)
        self.eps = float(eps)
        self._v = {}

    def step(self, params: dict, grads: dict, scale: float = 1.0):
        for k, g in grads.items():
            if k not in params:
                continue
            v = self._v.get(k)
            if v is None:
                v = np.zeros_like(params[k])
            v = v + np.square(g)
            params[k][:] = params[k] - (self.lr * scale) * g / np.sqrt(v + self.eps)
            self._v[k] = v


class Adadelta(Optimizer):
    def __init__(self, lr: float = 1., rho: float = 0.9, eps: float = 1e-06):
        super().__init__(lr=lr)
        self.rho = float(rho)
        self.eps = float(eps)
        self._m = {}
        self._v = {}

    def step(self, params: dict, grads: dict, scale: float = 1.0):
        for k, g in grads.items():
            if k not in params:
                continue
            m = self._m.get(k)
            v = self._v.get(k)
            if m is None:
                m = np.zeros_like(params[k])
            if v is None:
                v = np.zeros_like(params[k])
            v = self.rho * v + (1. - self.rho) * np.square(g)
            delta = np.sqrt(m + self.eps) / np.sqrt(v + self.eps) * g
            params[k][:] = params[k] - (self.lr * scale) * delta
            m = self.rho * m + (1. - self.rho) * np.square(delta)
            self._m[k] = m
            self._v[k] = v


class RMSProp(Optimizer):
    def __init__(self, lr: float = 0.01, alpha: float = 0.99, eps: float = 1e-08):
        super().__init__(lr=lr)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self._v = {}

    def step(self, params: dict, grads: dict, scale: float = 1.0):
        for k, g in grads.items():
            if k not in params:
                continue
            v = self._v.get(k)
            if v is None:
                v = np.zeros_like(params[k])
            v = self.alpha * v + (1. - self.alpha) * np.square(g)
            params[k][:] = params[k] - (self.lr * scale) * g / np.sqrt(v + self.eps)
            self._v[k] = v


class Adam(Optimizer):
    def __init__(self, lr: float = 0.01, betas: tuple = (0.9, 0.999), eps: float = 1e-08):
        super().__init__(lr=lr)
        self.betas = betas
        self.eps = float(eps)
        self._m = {}
        self._v = {}
        self._t = {}

    def step(self, params: dict, grads: dict, scale: float = 1.0):
        b1, b2 = self.betas
        for k, g in grads.items():
            if k not in params:
                continue
            m = self._m.get(k)
            v = self._v.get(k)
            t = self._t.get(k, 0) + 1
            if m is None:
                m = np.zeros_like(params[k])
            if v is None:
                v = np.zeros_like(params[k])
            m = b1 * m + (1. - b1) * g
            v = b2 * v + (1. - b2) * np.square(g)
            m_hat = m / (1. - b1**t)
            v_hat = v / (1. - b2**t)
            params[k][:] = params[k] - (self.lr * scale) * m_hat / (np.sqrt(v_hat) + self.eps)
            self._m[k] = m
            self._v[k] = v
            self._t[k] = t


class AdaMax(Optimizer):
    def __init__(self, lr: float = 0.01, betas: tuple = (0.9, 0.999), eps: float = 1e-08):
        super().__init__(lr=lr)
        self.betas = betas
        self.eps = float(eps)
        self._m = {}
        self._v = {}

    def step(self, params: dict, grads: dict, scale: float = 1.0):
        b1, b2 = self.betas
        for k, g in grads.items():
            if k not in params:
                continue
            m = self._m.get(k)
            v = self._v.get(k)
            if m is None:
                m = np.zeros_like(params[k])
            if v is None:
                v = np.zeros_like(params[k])
            m = b1 * m + (1. - b1) * g
            v = np.maximum(b2 * v, np.abs(g))
            m_hat = m / (1. - b1)
            params[k][:] = params[k] - (self.lr * scale) * m_hat / (v + self.eps)
            self._m[k] = m
            self._v[k] = v