"""Activation functions for ANN models.

Author: Nirmal Parmar

Notes:
- Includes class-based layers for easy use with `Sequential`.
- Also exposes functional helpers for direct tensor manipulation.
"""

import numpy as np

from ..layers import Layer
from ..tensor import Tensor


def relu(x: Tensor) -> Tensor:
    out_data = np.maximum(0.0, x.data)
    out = Tensor(out_data, requires_grad=x.requires_grad, device=x.device, dtype=x.data.dtype)
    out._prev = {x}

    def _backward():
        if out.grad is None:
            return
        if x.requires_grad:
            x.grad += out.grad * (x.data > 0.0)

    out._backward_fn = _backward
    return out


def leaky_relu(x: Tensor, negative_slope: float = 0.01) -> Tensor:
    out_data = np.where(x.data > 0.0, x.data, negative_slope * x.data)
    out = Tensor(out_data, requires_grad=x.requires_grad, device=x.device, dtype=x.data.dtype)
    out._prev = {x}

    def _backward():
        if out.grad is None:
            return
        if x.requires_grad:
            grad_scale = np.where(x.data > 0.0, 1.0, negative_slope)
            x.grad += out.grad * grad_scale

    out._backward_fn = _backward
    return out


def sigmoid(x: Tensor) -> Tensor:
    out_data = 1.0 / (1.0 + np.exp(-x.data))
    out = Tensor(out_data, requires_grad=x.requires_grad, device=x.device, dtype=x.data.dtype)
    out._prev = {x}

    def _backward():
        if out.grad is None:
            return
        if x.requires_grad:
            x.grad += out.grad * out_data * (1.0 - out_data)

    out._backward_fn = _backward
    return out


def tanh(x: Tensor) -> Tensor:
    out_data = np.tanh(x.data)
    out = Tensor(out_data, requires_grad=x.requires_grad, device=x.device, dtype=x.data.dtype)
    out._prev = {x}

    def _backward():
        if out.grad is None:
            return
        if x.requires_grad:
            x.grad += out.grad * (1.0 - out_data**2)

    out._backward_fn = _backward
    return out


def gelu(x: Tensor) -> Tensor:
    coeff = np.sqrt(2.0 / np.pi)
    x_cube = x.data**3
    tanh_arg = coeff * (x.data + 0.044715 * x_cube)
    tanh_val = np.tanh(tanh_arg)
    out_data = 0.5 * x.data * (1.0 + tanh_val)

    out = Tensor(out_data, requires_grad=x.requires_grad, device=x.device, dtype=x.data.dtype)
    out._prev = {x}

    def _backward():
        if out.grad is None:
            return
        if x.requires_grad:
            dtanh = 1.0 - tanh_val**2
            inner_grad = coeff * (1.0 + 3.0 * 0.044715 * x.data**2)
            grad = 0.5 * (1.0 + tanh_val) + 0.5 * x.data * dtanh * inner_grad
            x.grad += out.grad * grad

    out._backward_fn = _backward
    return out


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    shifted = x.data - np.max(x.data, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    probs = exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    out = Tensor(probs, requires_grad=x.requires_grad, device=x.device, dtype=x.data.dtype)
    out._prev = {x}

    def _backward():
        if out.grad is None:
            return
        if x.requires_grad:
            dot = np.sum(out.grad * probs, axis=axis, keepdims=True)
            x.grad += probs * (out.grad - dot)

    out._backward_fn = _backward
    return out


class ReLU(Layer):
    def forward(self, inputs: Tensor) -> Tensor:
        return relu(inputs)


class LeakyReLU(Layer):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = float(negative_slope)

    def forward(self, inputs: Tensor) -> Tensor:
        return leaky_relu(inputs, self.negative_slope)


class Sigmoid(Layer):
    def forward(self, inputs: Tensor) -> Tensor:
        return sigmoid(inputs)


class Tanh(Layer):
    def forward(self, inputs: Tensor) -> Tensor:
        return tanh(inputs)


class GELU(Layer):
    def forward(self, inputs: Tensor) -> Tensor:
        return gelu(inputs)


class Softmax(Layer):
    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def forward(self, inputs: Tensor) -> Tensor:
        return softmax(inputs, axis=self.axis)


class Identity(Layer):
    def forward(self, inputs: Tensor) -> Tensor:
        return inputs
