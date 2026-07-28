import numpy as np

class Optimizer:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params_and_grads):
        raise NotImplementedError
    
class SGD(Optimizer):
    """Vanilla stochastic gradient descent, with optional L2 weight decay."""
    def __init__(self, lr=0.01, weight_decay=0.0):
        super().__init__(lr)
        self.weight_decay= weight_decay

    def step(self, params_and_grads):
        for param, grad in params_and_grads:
            if self.weight_decay:
                grad= grad + self.weight_decay * param
            param -= self.lr * grad

class Momentum(Optimizer):
    """SGD with classical momentum: v = mu*v - lr*grad; param += v."""
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__(lr)
        self.momentum = momentum
        self.velocities = {}

    def step(self, params_and_grads):
        for param, grad in params_and_grads:
            key= id(param)
            if key not in self.velocities:
                self.velocities[key]= np.zeros_like(param)
            v = self.velocities[key]
            v[:]= self.momentum * v- self.lr * grad
            param += v
            
class RMSProp(Optimizer):
    """Adapts the learning rate per-parameter using a decaying average of squared grads."""
    def __init__(self, lr=0.001, decay_rate=0.9, eps=1e-8):
        super().__init__(lr)
        self.decay_rate= decay_rate
        self.eps = eps
        self.cache = {}

    def step(self, params_and_grads):
        for param, grad in params_and_grads:
            key= id(param)
            if key not in self.cache:
                self.cache[key]= np.zeros_like(param)
            cache = self.cache[key]
            cache[:]= self.decay_rate * cache + (1- self.decay_rate) * grad ** 2
            param -= self.lr * grad / (np.sqrt(cache) + self.eps)

class Adam(Optimizer):
    """
    scaling (second moment), plus bias correction for early steps.
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = {}

    def step(self, params_and_grads):
        for param, grad in params_and_grads:
            key= id(param)
            if key not in self.m:
                self.m[key]= np.zeros_like(param)
                self.v[key]= np.zeros_like(param)
                self.t[key]= 0
            self.t[key] += 1
            t = self.t[key]
            m, v = self.m[key], self.v[key]
            m[:]= self.beta1 * m + (1- self.beta1) * grad
            v[:]= self.beta2 * v + (1- self.beta2) * grad ** 2
            m_hat= m / (1- self.beta1 ** t)
            v_hat= v / (1- self.beta2 ** t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)