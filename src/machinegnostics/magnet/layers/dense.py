"""
Dense layer (neuron_type removed).

Notes:
- The prior `neuron_type` argument (E/Q) did not influence any
	gnostic-specific calculations. It was only used to choose default
	activations and a slightly different weight init scale.
- This layer now ignores `neuron_type` entirely. If legacy code passes
	it, a warning is logged and it's ignored.
- Default activation: ReLU. For non-ReLU behaviors (e.g., quadratic),
	pass `activation='quadratic'` explicitly.
"""
import numpy as np
import logging
from .base import BaseLayer
from machinegnostics.magnet.activations import get_activation
from machinegnostics.magcal.util.logging import get_logger

class Dense(BaseLayer):
	def __init__(self,
			  units: int,
			  activation: str = None,
			  **kwargs):
		# Ignore legacy neuron_type if provided
		if 'neuron_type' in kwargs:
			logger = get_logger(self.__class__.__name__, logging.WARNING)
		super().__init__(name=f"Dense({units})")
		self.units = units
		# default activation: relu
		activation = activation or 'relu'
		self.activation_name = activation
		self.activation, self.activation_grad = get_activation(activation)
		self.logger = get_logger(self.__class__.__name__, logging.WARNING)

	def build(self, input_shape):
		in_features = input_shape[-1]
		# Xavier/He-like init adjusted by activation type
		if (self.activation_name or '').lower().startswith('quadratic'):
			scale = np.sqrt(1.0 / in_features)
		else:
			scale = np.sqrt(2.0 / in_features)
		W = np.random.randn(in_features, self.units) * scale
		b = np.zeros((1, self.units))
		self.params = {
			'W': W,
			'b': b
		}
		self.built = True
		self.input_shape = input_shape
		self.output_shape = (input_shape[0], self.units)

	def forward(self, x):
		self.x = x  # cache
		W, b = self.params['W'], self.params['b']
		z = x @ W + b
		self.z = z
		return self.activation(z)

	def backward(self, grad_out):
		# grad_out: gradient wrt activation output
		dz = grad_out * self.activation_grad(self.z)
		dW = self.x.T @ dz
		db = np.sum(dz, axis=0, keepdims=True)
		dx = dz @ self.params['W'].T
		self.grads = {
			'W': dW,
			'b': db
		}
		return dx
