import numpy as np
from .layer import Layer
from .initializer import Zeros, XavierUniform
from .engine import GnosticEngine

class iDense(Layer):
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
        self.params["W"] = weight_init((in_features, out_features))
        self.params["b"] = bias_init((out_features,))
        self.grads["W"] = np.zeros_like(self.params["W"])
        self.grads["b"] = np.zeros_like(self.params["b"])
        self.S = S  # Store the scale parameter S
        self.gnostic_engine = GnosticEngine(S=self.S, verbose=False)  # Initialize the GnosticEngine

    def forward(self, x, training=True):
        self.input = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, grad_output):
        # gnostic weight estimation from gradient output
        self.wg = self.gnostic_engine._get_gnostic_i_weights(grad_output, scale_param=self.S)
        # grad_output: (batch, out_features)
        grad_output_weighted = np.multiply(grad_output, self.wg)
        self.grads["W"] = self.input.T @ grad_output_weighted
        self.grads["b"] = grad_output_weighted.sum(axis=0)
        grad_input = grad_output_weighted @ self.params["W"].T
        return grad_input


class iwDense(Layer):
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
        self.params["W"] = weight_init((in_features, out_features))
        self.params["b"] = bias_init((out_features,))
        self.grads["W"] = np.zeros_like(self.params["W"])
        self.grads["b"] = np.zeros_like(self.params["b"])
        self.S = S  # Store the scale parameter S
        self.gnostic_engine = GnosticEngine(S=self.S, verbose=False)  # Initialize the GnosticEngine

    def forward(self, x, training=True):
        self.input = x
    
        input_median = np.median(x, axis=0, keepdims=True)
        input_centered = x - input_median
        input_z = np.exp(input_centered)
    
        self.wg = self.gnostic_engine._get_gnostic_i_weights(input_z, scale_param=self.S)
    
        if self.wg.shape != x.shape:
            self.wg = np.broadcast_to(self.wg, x.shape)
    
        x_weighted = x * self.wg
        self.input = x_weighted
        return x_weighted @ self.params["W"] + self.params["b"]

    def backward(self, grad_output):
        # grad_output: (batch, out_features)
        self.grads["W"] = self.input.T @ grad_output
        self.grads["b"] = grad_output.sum(axis=0)
        grad_input = grad_output @ self.params["W"].T
        return grad_input