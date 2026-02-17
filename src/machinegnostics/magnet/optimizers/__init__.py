"""
Optimizers for Machine Gnostics Neural Networks.
Supports sample-weighted gradient updates.
"""
import numpy as np

class Optimizer:
    def __init__(self, lr: float = 0.001):
        self.lr = lr

    def step(self, params, grads):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, lr: float = 0.01, momentum: float = 0.0):
        super().__init__(lr)
        self.momentum = momentum
        self.v = {}

    def step(self, params, grads):
        for k in params:
            if k not in self.v:
                self.v[k] = np.zeros_like(params[k])
            self.v[k] = self.momentum * self.v[k] - self.lr * grads[k]
            params[k] += self.v[k]

class RMSProp(Optimizer):
    def __init__(self, lr: float = 0.001, beta: float = 0.9, eps: float = 1e-8):
        super().__init__(lr)
        self.beta = beta
        self.eps = eps
        self.s = {}

    def step(self, params, grads):
        for k in params:
            if k not in self.s:
                self.s[k] = np.zeros_like(params[k])
            self.s[k] = self.beta * self.s[k] + (1 - self.beta) * (grads[k] ** 2)
            params[k] -= self.lr * grads[k] / (np.sqrt(self.s[k]) + self.eps)

class Adam(Optimizer):
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            if k not in self.m:
                self.m[k] = np.zeros_like(params[k])
                self.v[k] = np.zeros_like(params[k])
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (grads[k] ** 2)
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
