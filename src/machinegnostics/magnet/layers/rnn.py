"""
Simple RNN layer for sequence modeling.
Assumes input shape (batch, time, features).
"""
import numpy as np
from .base import BaseLayer
from machinegnostics.magnet.activations import get_activation

class SimpleRNN(BaseLayer):
    def __init__(self, units: int, activation: str = 'tanh', neuron_type: str = 'E', return_sequences: bool = False):
        super().__init__(name=f"SimpleRNN({units})", neuron_type=neuron_type)
        self.units = units
        self.return_sequences = return_sequences
        self.activation_name = activation
        self.activation, self.activation_grad = get_activation(activation)

    def build(self, input_shape):
        # input_shape: (batch, time, features)
        _, T, F = input_shape
        scale = np.sqrt(1.0 / F)
        self.params = {
            'Wx': np.random.randn(F, self.units) * scale,
            'Wh': np.random.randn(self.units, self.units) * scale,
            'b': np.zeros((1, self.units))
        }
        self.built = True
        self.input_shape = input_shape
        self.output_shape = (input_shape[0], T if self.return_sequences else 1, self.units)

    def forward(self, x):
        self.x = x
        batch, T, F = x.shape
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        h = np.zeros((batch, self.units))
        self.cache = []
        outputs = []
        for t in range(T):
            z = x[:, t, :] @ Wx + h @ Wh + b
            h = self.activation(z)
            self.cache.append((z, h))
            outputs.append(h)
        self.h_last = h
        if self.return_sequences:
            return np.stack(outputs, axis=1)
        else:
            return h

    def backward(self, grad_out):
        batch, T, F = self.x.shape
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        dWx = np.zeros_like(Wx)
        dWh = np.zeros_like(Wh)
        db = np.zeros_like(b)
        dx = np.zeros_like(self.x)
        dh_next = grad_out if not self.return_sequences else grad_out[:, -1, :]
        for t in reversed(range(T)):
            z, h = self.cache[t]
            dh = dh_next
            dz = dh * self.activation_grad(z)
            dWx += self.x[:, t, :].T @ dz
            if t > 0:
                h_prev = self.cache[t-1][1]
            else:
                h_prev = np.zeros((batch, self.units))
            dWh += h_prev.T @ dz
            db += np.sum(dz, axis=0, keepdims=True)
            dx[:, t, :] = dz @ Wx.T
            dh_next = dz @ Wh.T
        self.grads = {'Wx': dWx, 'Wh': dWh, 'b': db}
        return dx
