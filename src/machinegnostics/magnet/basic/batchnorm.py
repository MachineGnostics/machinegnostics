import numpy as np
from .layer import Layer
from .engine import GnosticEngine

class BatchNorm(Layer):
    """
    Batch normalization over the feature axis for (N, features) input.
    Maintains running mean/var for use at inference time.
    """
    def __init__(self, num_features, momentum=0.9, eps=1e-5, name=None):
        super().__init__(name)
        self.momentum = momentum
        self.eps = eps
        self.params["gamma"]= np.ones(num_features)
        self.params["beta"]= np.zeros(num_features)
        self.grads["gamma"]= np.zeros(num_features)
        self.grads["beta"]= np.zeros(num_features)
        self.running_mean= np.zeros(num_features)
        self.running_var= np.ones(num_features)

    def forward(self, x, training=True):
        if training:
            batch_mean= x.mean(axis=0)
            batch_var= x.var(axis=0)
            self.x_centered= x- batch_mean
            self.std_inv= 1.0 / np.sqrt(batch_var + self.eps)
            self.x_norm = self.x_centered * self.std_inv
            self.running_mean= self.momentum * self.running_mean + (1- self.momentum) * batch_mean
            self.running_var= self.momentum * self.running_var + (1- self.momentum) * batch_var
        else:
            self.x_norm = (x- self.running_mean) / np.sqrt(self.running_var + self.eps)
        return self.params["gamma"] * self.x_norm + self.params["beta"]

    def backward(self, grad_output):
        N = grad_output.shape[0]
        gamma = self.params["gamma"]
        self.grads["gamma"]= np.sum(grad_output * self.x_norm, axis=0)
        self.grads["beta"]= np.sum(grad_output, axis=0)
        dx_norm= grad_output * gamma
        dvar_term= dx_norm * self.x_centered * (-0.5) * self.std_inv ** 3
        dvar = np.sum(dvar_term, axis=0)
        dmean = np.sum(dx_norm * -self.std_inv, axis=0) + dvar * np.mean(-2.0 * self.x_centered, axis=0)
        grad_input= (dx_norm * self.std_inv) + (dvar * 2.0 * self.x_centered / N) + (dmean / N)
        return grad_input

class GnosticBatchNorm(Layer):
    """
    Batch normalization over the feature axis for (N, features) input.
    Maintains running mean/var for use at inference time.
    """
    def __init__(self, num_features, momentum=0.9, eps=1e-5, name=None):
        super().__init__(name)
        self.momentum = momentum
        self.eps = eps
        self.params["gamma"]= np.ones(num_features)
        self.params["beta"]= np.zeros(num_features)
        self.grads["gamma"]= np.zeros(num_features)
        self.grads["beta"]= np.zeros(num_features)
        self.running_mean= np.zeros(num_features)
        self.running_var= np.ones(num_features)

    def forward(self, x, training=True):
        if training:
            batch_mean= x.mean(axis=0)
            batch_var= x.var(axis=0)
            self.x_centered= x- batch_mean
            x_centered_z = np.exp(self.x_centered)
            self.gw = GnosticEngine()._get_gnostic_i_weights(x_centered_z, scale_param=2.0)
            self.std_inv= 1.0 / np.sqrt(batch_var + self.eps)
            self.x_norm = self.x_centered * self.std_inv * self.gw
            self.running_mean= self.momentum * self.running_mean + (1- self.momentum) * batch_mean
            self.running_var= self.momentum * self.running_var + (1- self.momentum) * batch_var
        else:
            self.x_norm = (x- self.running_mean) / np.sqrt(self.running_var + self.eps)
        return self.params["gamma"] * self.x_norm + self.params["beta"]

    def backward(self, grad_output):
        N = grad_output.shape[0]
        gamma = self.params["gamma"]
        self.grads["gamma"]= np.sum(grad_output * self.x_norm, axis=0)
        self.grads["beta"]= np.sum(grad_output, axis=0)
        dx_norm= grad_output * gamma
        dvar_term= dx_norm * self.x_centered * (-0.5) * self.std_inv ** 3
        dvar = np.sum(dvar_term, axis=0)
        dmean = np.sum(dx_norm * -self.std_inv, axis=0) + dvar * np.mean(-2.0 * self.x_centered, axis=0)
        grad_input= (dx_norm * self.std_inv) + (dvar * 2.0 * self.x_centered / N) + (dmean / N)
        return grad_input