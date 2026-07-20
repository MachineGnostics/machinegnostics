"""
Gnostic Activation Functions

ManGo - Machine Gnostics Library
Copyright (C) 2026 Nirmal Parmar
"""

import logging

import numpy as np

from machinegnostics.magcal.util.logging import get_logger


class ActivationFunctions:
	"""
	### Activation Functions Class

	This class provides regular activation functions for neuron models.
	It keeps activation selection, validation, and logging in one place so
	callers can request a specific activation by name.

	Notes:
	- Inputs are converted to `numpy` arrays before evaluation.
	- Supported activation types include `step`, `linear`, `sigmoid`, `tanh`,
	  `relu`, `leaky_relu`, `elu`, `softplus`, `swish`, `gelu`, `mish`, and
	  `softmax`.
	"""

	def __init__(self, verbose: bool = False):
		self.verbose = verbose
		self.logger = get_logger(
			self.__class__.__name__,
			logging.DEBUG if self.verbose else logging.WARNING,
		)
		self.logger.info(f"{self.__class__.__name__} initialized.")

	def _as_array(self, z):
		"""Convert the input to a floating-point numpy array."""
		return np.asarray(z, dtype=float)

	def linear(self, z):
		"""
		Apply the linear activation.

		Notes:
		- This is the identity transformation.
		"""
		self.logger.info("Computing linear activation.")
		return self._as_array(z)

	def sigmoid(self, z):
		"""
		Apply the sigmoid activation.

		Notes:
		- Output is bounded between 0 and 1.
		"""
		self.logger.info("Computing sigmoid activation.")
		z = self._as_array(z)
		return 1 / (1 + np.exp(-z))

	def tanh(self, z):
		"""
		Apply the hyperbolic tangent activation.

		Notes:
		- Output is bounded between -1 and 1.
		"""
		self.logger.info("Computing tanh activation.")
		z = self._as_array(z)
		return np.tanh(z)

	def relu(self, z):
		"""
		Apply the rectified linear unit activation.

		Notes:
		- Negative values are clipped to zero.
		"""
		self.logger.info("Computing relu activation.")
		z = self._as_array(z)
		return np.maximum(0, z)

	def step(self, z):
		"""
		Apply the step activation.

		Notes:
		- Values greater than or equal to zero map to 1, otherwise 0.
		"""
		self.logger.info("Computing step activation.")
		z = self._as_array(z)
		return np.where(z >= 0, 1, 0)

	def leaky_relu(self, z, alpha: float = 0.01):
		"""
		Apply the leaky ReLU activation.

		Notes:
		- Negative values are scaled by `alpha` instead of being clipped to zero.
		"""
		self.logger.info("Computing leaky_relu activation.")
		z = self._as_array(z)
		return np.where(z > 0, z, alpha * z)

	def elu(self, z, alpha: float = 0.01):
		"""
		Apply the exponential linear unit activation.

		Notes:
		- Positive values pass through unchanged.
		- Negative values use an exponential decay controlled by `alpha`.
		"""
		self.logger.info("Computing elu activation.")
		z = self._as_array(z)
		return np.where(z > 0, z, alpha * (np.exp(z) - 1))

	def softplus(self, z):
		"""
		Apply the softplus activation.

		Notes:
		- This is a smooth approximation of ReLU.
		- Implemented with `log1p` for better numerical stability.
		"""
		self.logger.info("Computing softplus activation.")
		z = self._as_array(z)
		return np.log1p(np.exp(z))

	def swish(self, z):
		"""
		Apply the swish activation.

		Notes:
		- Swish is defined as `z * sigmoid(z)`.
		"""
		self.logger.info("Computing swish activation.")
		z = self._as_array(z)
		return z / (1 + np.exp(-z))

	def gelu(self, z):
		"""
		Apply the Gaussian error linear unit activation.

		Notes:
		- Uses the common tanh-based approximation.
		"""
		self.logger.info("Computing gelu activation.")
		z = self._as_array(z)
		return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * np.power(z, 3))))

	def mish(self, z):
		"""
		Apply the mish activation.

		Notes:
		- Mish is defined as `z * tanh(softplus(z))`.
		"""
		self.logger.info("Computing mish activation.")
		z = self._as_array(z)
		return z * np.tanh(np.log1p(np.exp(z)))

	def softmax(self, z):
		"""
		Apply the softmax activation.

		Notes:
		- The last axis is normalized.
		- A max-subtraction is used for numerical stability.
		"""
		self.logger.info("Computing softmax activation.")
		z = self._as_array(z)
		if z.ndim == 0:
			return np.array(1.0)
		shifted = z - np.max(z, axis=-1, keepdims=True)
		exp_values = np.exp(shifted)
		return exp_values / np.sum(exp_values, axis=-1, keepdims=True)

	def activate(self, z, activation_type: str = 'linear'):
		"""
		Dispatch to a specific activation function.

		Parameters:
		- z: Input values.
		- activation_type: Activation name.

		Returns:
		- activation: Activated values.
		"""
		self.logger.info(f"Dispatching activation type: {activation_type}.")

		if activation_type == 'linear':
			return self.linear(z)
		elif activation_type == 'step':
			return self.step(z)
		elif activation_type == 'sigmoid':
			return self.sigmoid(z)
		elif activation_type == 'tanh':
			return self.tanh(z)
		elif activation_type == 'relu':
			return self.relu(z)
		elif activation_type == 'leaky_relu':
			return self.leaky_relu(z)
		elif activation_type == 'elu':
			return self.elu(z)
		elif activation_type == 'softplus':
			return self.softplus(z)
		elif activation_type == 'swish':
			return self.swish(z)
		elif activation_type == 'gelu':
			return self.gelu(z)
		elif activation_type == 'mish':
			return self.mish(z)
		elif activation_type == 'softmax':
			return self.softmax(z)
		else:
			raise ValueError(f"Unsupported activation_type: {activation_type}")

	def available_activations(self):
		"""Return the supported activation names."""
		return ('step', 'linear', 'sigmoid', 'tanh', 'relu', 'leaky_relu', 'elu', 'softplus', 'swish', 'gelu', 'mish', 'softmax')

	def __repr__(self):
		return f"ActivationFunctions(verbose={self.verbose})"
