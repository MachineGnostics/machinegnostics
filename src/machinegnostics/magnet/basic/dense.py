import numpy as np
from .layer import Layer
from .initializer import Zeros, XavierUniform

class Dense(Layer):
    """
    Fully connected layer: y = x @ W + b
    x: (batch, in_features)
    W: (in_features, out_features)
    b: (out_features,)
    """
    def __init__(self, in_features, out_features,
                 weight_init=None, bias_init=None, name=None):
        super().__init__(name)
        weight_init = weight_init or XavierUniform()
        bias_init = bias_init or Zeros()
        self.params["W"] = weight_init((in_features, out_features))
        self.params["b"] = bias_init((out_features,))
        self.grads["W"] = np.zeros_like(self.params["W"])
        self.grads["b"] = np.zeros_like(self.params["b"])

    def forward(self, x, training=True):
        self.input = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, grad_output):
        # grad_output: (batch, out_features)
        self.grads["W"] = self.input.T @ grad_output
        self.grads["b"] = grad_output.sum(axis=0)
        grad_input = grad_output @ self.params["W"].T
        return grad_input