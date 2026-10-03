"""Dense (fully connected) layer for MAGNET.

Developer note
--------------
Author: Nirmal Parmar

``Dense`` implements the affine transform ``y = x @ W + b``. The layer keeps
its parameters as MAGNET tensors so gradients flow through the shared tensor
autograd engine while still exposing an optional manual backward path for
inspection and compatibility.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..core.tensor import Tensor
from ..initializers import XavierUniform, Zeros, get_initializer
from .base import Layer


class Dense(Layer):
	"""Fully connected linear layer with trainable weights and bias.

	The layer computes the affine transform ``y = x @ W + b``. It is the core
	building block for most MAGNET networks because it turns input features into
	a learned representation using a trainable weight matrix and bias vector.

	Unlike a standalone NumPy implementation, ``Dense`` stores its parameters as
	``Tensor`` objects so gradients can flow through MAGNET's autograd engine.
	The layer also exposes a manual ``backward`` hook so you can inspect or
	override gradient logic when you need custom behavior.
	"""

	def __init__(
		self,
		in_features=None,
		out_features=None,
		weight_init=None,
		bias_init=None,
		name=None,
		verbose: bool = False,
		backward_fn: Callable[["Dense", Any], Any] | None = None,
		*,
		n_in=None,
		n_out=None,
		use_bias: bool = True,
		kernel_initializer=None,
		bias_initializer=None,
	):
		"""Create a dense layer.

		Parameters
		----------
		in_features:
			Input dimensionality.
		out_features:
			Number of output units.
		weight_init:
			Backward-compatible alias for ``kernel_initializer``.
		bias_init:
			Backward-compatible alias for ``bias_initializer``.
		name:
			Optional layer name.
		verbose:
			Enable debug logging for the layer instance.
		backward_fn:
			Optional custom backward callback.
		n_in, n_out:
			Alternative parameter names kept for compatibility with the phase spec.
		use_bias:
			Whether to include a trainable bias vector.
		kernel_initializer, bias_initializer:
			Preferred initializer names for weights and biases.
		"""
		super().__init__(name, verbose=verbose)
		in_features = in_features if in_features is not None else n_in
		out_features = out_features if out_features is not None else n_out
		if in_features is None or out_features is None:
			raise ValueError("Dense requires in_features/out_features or n_in/n_out")

		weight_init = self._resolve_initializer(kernel_initializer if kernel_initializer is not None else weight_init)
		bias_init = self._resolve_initializer(bias_initializer if bias_initializer is not None else bias_init, default=Zeros())

		self.in_features = int(in_features)
		self.out_features = int(out_features)
		self.n_in = self.in_features
		self.n_out = self.out_features
		self.use_bias = bool(use_bias)
		self.trainable = True

		self.params["W"] = self._ensure_parameter_tensor(
			weight_init((self.in_features, self.out_features)),
			f"{self.name}_W",
		)
		self.W = self.params["W"]
		self.grads["W"] = None

		if self.use_bias:
			self.params["b"] = self._ensure_parameter_tensor(
				bias_init((self.out_features,)),
				f"{self.name}_b",
			)
			self.b = self.params["b"]
			self.grads["b"] = None
		else:
			self.b = None

		self.backward_fn = backward_fn
		self.logger.debug(
			"Dense initialized with in_features=%s, out_features=%s, use_bias=%s.",
			self.in_features,
			self.out_features,
			self.use_bias,
		)

	@staticmethod
	def _resolve_initializer(initializer, default=None):
		"""Resolve initializer aliases used across the refactor."""
		if initializer is None:
			return default if default is not None else XavierUniform(seed=42)
		if isinstance(initializer, str):
			alias_map = {
				"glorot_uniform": "xavieruniform",
				"glorot-uniform": "xavieruniform",
				"random_normal": "randomnormal",
				"he_normal": "henormal",
			}
			return get_initializer(alias_map.get(initializer, initializer))
		return initializer

	@staticmethod
	def _ensure_parameter_tensor(value, name: str) -> Tensor:
		"""Wrap raw initializer output in a trainable tensor when needed."""
		tensor = value if isinstance(value, Tensor) else Tensor(value)
		tensor.name = name
		tensor.requires_grad = True
		return tensor

	def forward(self, x, training=True):
		"""Apply the affine transform to the input tensor."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		if x.ndim == 1:
			x = x.reshape(1, -1)
		if x.shape[-1] != self.in_features:
			raise ValueError(f"Dense expected input with {self.in_features} features, got shape {x.shape}")
		self.input = x
		self.logger.debug("Running dense forward pass with input shape %s.", x.shape)
		output = x @ self.W
		if self.use_bias and self.b is not None:
			output = output + self.b
		return output

	def backward(self, grad_output):
		"""Compute or customize the dense backward pass."""
		if self.backward_fn is not None:
			self.logger.debug("Dense.backward delegated to custom backward_fn.")
			return self.backward_fn(self, grad_output)

		if not hasattr(self, "input") or self.input is None:
			raise RuntimeError("Dense.backward requires a prior forward pass")

		grad_output_tensor = grad_output if isinstance(grad_output, Tensor) else Tensor(grad_output)
		input_data = self.input.data
		grad_output_data = grad_output_tensor.data
		weight_data = self.W.data

		grad_w = input_data.T @ grad_output_data
		grad_input = grad_output_data @ weight_data.T

		self.grads["W"] = grad_w
		self.W.grad = grad_w
		if self.use_bias and self.b is not None:
			grad_b = np.asarray(grad_output_data.sum(axis=0), dtype=np.float64)
			self.grads["b"] = grad_b
			self.b.grad = grad_b
		self.logger.debug(
			"Dense.backward computed gradients with input shape %s and grad_output shape %s.",
			self.input.shape,
			grad_output_tensor.shape,
		)
		return Tensor(grad_input)
