"""
Activation functions for Machine Gnostics Neural Networks.
"""
import numpy as np

class Activation:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, x):
        return activate(self.name, x)

    def grad(self, x):
        return activation_grad(self.name, x)


def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(x.dtype)

def sigmoid(x):
    # numerically stable sigmoid
    pos_mask = x >= 0
    neg_mask = ~pos_mask
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1.0 + z)

def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_grad(x):
    t = tanh(x)
    return 1 - t**2

def linear(x):
    return x

def linear_grad(x):
    return np.ones_like(x)

def softmax(x):
    # x: (batch, features)
    x_shift = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x_shift)
    return e / np.sum(e, axis=-1, keepdims=True)

# Quadratic activation for Q-neuron

def quadratic(x):
    return x**2

def quadratic_grad(x):
    return 2 * x

_ACTS = {
    'relu': (relu, relu_grad),
    'sigmoid': (sigmoid, sigmoid_grad),
    'tanh': (tanh, tanh_grad),
    'linear': (linear, linear_grad),
    'softmax': (softmax, None),
    'quadratic': (quadratic, quadratic_grad),
}


def get_activation(name: str):
    if name not in _ACTS:
        raise ValueError(f"Unknown activation: {name}")
    f, g = _ACTS[name]
    return f, g


def activate(name: str, x):
    f, _ = get_activation(name)
    return f(x)


def activation_grad(name: str, x):
    _, g = get_activation(name)
    if g is None:
        raise ValueError(f"Activation {name} has no gradient (e.g., use with appropriate loss)")
    return g(x)
