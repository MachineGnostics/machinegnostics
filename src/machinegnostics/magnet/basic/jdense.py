import numpy as np
from .layer import Layer
from .initializer import Zeros, XavierUniform
from .engine import GnosticEngine

class jDense(Layer):
    """
    Fully connected layer: y = x @ W + b
    x: (batch, in_features)
    W: (in_features, out_features)
    b: (out_features,)
    """
    def __init__(self,
                 in_features, 
                 out_features,
                 weight_init=None, 
                 bias_init=None, 
                 name=None,
                 S: float | str = 2.0):
        super().__init__(name)
        weight_init = weight_init or XavierUniform()
        bias_init = bias_init or Zeros()
        self.S = S  # Store the scale parameter S
        self.params["W"] = weight_init((in_features, out_features))
        self.params["b"] = bias_init((out_features,))
        self.grads["W"] = np.zeros_like(self.params["W"])
        self.grads["b"] = np.zeros_like(self.params["b"])
        self.gnostic_engine = GnosticEngine(S=self.S, verbose=False)  # Initialize the GnosticEngine

    def forward(self, x, training=True):
        self.input = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, grad_output):
        # gnostic weight estimation from gradient output
        self.wg = self.gnostic_engine._get_gnostic_j_weights(grad_output, scale_param=self.S)
        # grad_output: (batch, out_features)
        grad_output_weighted = np.multiply(grad_output, self.wg)
        self.grads["W"] = self.input.T @ grad_output_weighted
        self.grads["b"] = grad_output_weighted.sum(axis=0)
        grad_input = grad_output_weighted @ self.params["W"].T
        return grad_input