"""Autograd tensor primitive for MAGNET (Machine Gnostics Neural Networks).

Developer note
-------------
Author: Nirmal Parmar

This tensor is intentionally small and explicit: it supports the arithmetic,
matrix, and reduction operations needed by the MAGNET layers and losses.

Examples
--------
>>> import numpy as np
>>> from machinegnostics.magnet.tensor import Tensor
>>> x = Tensor(np.array([[1., 2.]]), requires_grad=True)
>>> w = Tensor(np.array([[3.], [4.]]), requires_grad=True)
>>> y = (x @ w).sum()
>>> y.backward()
>>> x.grad.shape
(1, 2)
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)


def _ensure_array(value) -> np.ndarray:
	if isinstance(value, Tensor):
		return value.data
	return np.asarray(value, dtype=np.float64)


def unbroadcast(gradient: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
	if gradient.shape == shape:
		return gradient
	while gradient.ndim > len(shape):
		gradient = gradient.sum(axis=0)
	for axis, size in enumerate(shape):
		if size == 1 and gradient.shape[axis] != 1:
			gradient = gradient.sum(axis=axis, keepdims=True)
	return gradient.reshape(shape)


class Tensor:
	"""A NumPy-backed tensor with reverse-mode automatic differentiation.

	Parameters
	----------
	data:
		Any array-like object that can be converted to ``numpy.ndarray``.
	requires_grad:
		Whether gradients should be tracked for this tensor.
	name:
		Optional display name used in debugging and inspection.
	"""
	__array_priority__ = 1000

	def __init__(self, data, requires_grad: bool = False, name: str | None = None):
		"""Create a tensor from raw data.

		Examples
		--------
		>>> from machinegnostics.magnet import Tensor
		>>> Tensor([1, 2, 3]).shape
		(3,)
		"""
		self.data = np.asarray(data, dtype=np.float64)
		self.requires_grad = requires_grad
		self.name = name
		self.grad: np.ndarray | None = None
		self._prev: set[Tensor] = set()
		self._backward: Callable[[], None] = lambda: None
		logger.debug("Tensor initialized with shape %s and requires_grad=%s.", self.data.shape, self.requires_grad)

	@staticmethod
	def _ensure_tensor(value) -> "Tensor":
		"""Convert a scalar or array-like object into a ``Tensor``."""
		return value if isinstance(value, Tensor) else Tensor(value)

	def _add_grad(self, gradient: np.ndarray) -> None:
		"""Accumulate a gradient contribution into ``self.grad``."""
		if not self.requires_grad:
			return
		gradient = np.asarray(gradient, dtype=np.float64)
		self.grad = gradient if self.grad is None else self.grad + gradient
		logger.debug("Accumulated gradient for tensor with shape %s.", self.data.shape)

	def zero_grad(self) -> None:
		"""Reset the stored gradient to zeros with the same shape as the data."""
		self.grad = np.zeros_like(self.data)
		logger.debug("Cleared tensor gradient for shape %s.", self.data.shape)

	def detach(self) -> "Tensor":
		"""Return a non-tracking copy of the tensor."""
		logger.debug("Detached tensor with shape %s.", self.data.shape)
		return Tensor(self.data.copy(), requires_grad=False, name=self.name)

	def backward(self, gradient=None) -> None:
		"""Backpropagate through the computation graph.

		If the tensor is scalar, the default upstream gradient is 1.0.
		"""
		if not self.requires_grad:
			return
		logger.debug("Running backward pass for tensor with shape %s.", self.data.shape)
		if gradient is None:
			if self.data.size != 1:
				raise ValueError("gradient must be provided for non-scalar tensors")
			gradient = np.ones_like(self.data)
		else:
			gradient = np.asarray(gradient, dtype=np.float64)

		topo: list[Tensor] = []
		visited: set[int] = set()

		def build(node: Tensor) -> None:
			if id(node) in visited:
				return
			visited.add(id(node))
			for parent in node._prev:
				build(parent)
			topo.append(node)

		build(self)
		self.grad = gradient if self.grad is None else self.grad + gradient
		for node in reversed(topo):
			node._backward()

	def _binary_op(self, other, op, grad_self, grad_other):
		other = Tensor._ensure_tensor(other)
		out = Tensor(op(self.data, other.data), requires_grad=self.requires_grad or other.requires_grad)
		out._prev = {self, other}

		def _backward() -> None:
			if out.grad is None:
				return
			if self.requires_grad:
				self._add_grad(unbroadcast(grad_self(out.grad, self.data, other.data), self.data.shape))
			if other.requires_grad:
				other._add_grad(unbroadcast(grad_other(out.grad, self.data, other.data), other.data.shape))

		out._backward = _backward
		return out

	def __add__(self, other):
		return self._binary_op(other, np.add, lambda g, x, y: g, lambda g, x, y: g)

	def __radd__(self, other):
		return self.__add__(other)

	def __sub__(self, other):
		return self._binary_op(other, np.subtract, lambda g, x, y: g, lambda g, x, y: -g)

	def __rsub__(self, other):
		other = Tensor._ensure_tensor(other)
		return other.__sub__(self)

	def __mul__(self, other):
		return self._binary_op(other, np.multiply, lambda g, x, y: g * y, lambda g, x, y: g * x)

	def __rmul__(self, other):
		return self.__mul__(other)

	def __truediv__(self, other):
		return self._binary_op(other, np.divide, lambda g, x, y: g / y, lambda g, x, y: -g * x / (y ** 2))

	def __rtruediv__(self, other):
		other = Tensor._ensure_tensor(other)
		return other.__truediv__(self)

	def __pow__(self, power):
		if isinstance(power, Tensor):
			raise TypeError("Tensor powers with Tensor exponents are not supported")
		out = Tensor(self.data ** power, requires_grad=self.requires_grad)
		out._prev = {self}

		def _backward() -> None:
			if self.requires_grad and out.grad is not None:
				self._add_grad(unbroadcast(out.grad * power * (self.data ** (power - 1)), self.data.shape))

		out._backward = _backward
		return out

	def __neg__(self):
		out = Tensor(-self.data, requires_grad=self.requires_grad)
		out._prev = {self}

		def _backward() -> None:
			if self.requires_grad and out.grad is not None:
				self._add_grad(-out.grad)

		out._backward = _backward
		return out

	def __matmul__(self, other):
		other = Tensor._ensure_tensor(other)
		out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)
		out._prev = {self, other}

		def _backward() -> None:
			if out.grad is None:
				return
			if self.requires_grad:
				self._add_grad(out.grad @ other.data.T)
			if other.requires_grad:
				other._add_grad(self.data.T @ out.grad)

		out._backward = _backward
		return out

	def sum(self, axis=None, keepdims: bool = False):
		"""Sum tensor elements with autograd support."""
		out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
		out._prev = {self}

		def _backward() -> None:
			if self.requires_grad and out.grad is not None:
				gradient = out.grad
				if axis is None:
					gradient = np.broadcast_to(gradient, self.data.shape)
				else:
					axes = axis if isinstance(axis, tuple) else (axis,)
					if not keepdims:
						for ax in sorted(axes):
							gradient = np.expand_dims(gradient, axis=ax)
					gradient = np.broadcast_to(gradient, self.data.shape)
				self._add_grad(gradient)

		out._backward = _backward
		return out

	def mean(self, axis=None, keepdims: bool = False):
		"""Compute the mean with gradient propagation."""
		if axis is None:
			count = self.data.size
		else:
			axes = axis if isinstance(axis, tuple) else (axis,)
			count = 1
			for ax in axes:
				count *= self.data.shape[ax]
		return self.sum(axis=axis, keepdims=keepdims) / count

	def reshape(self, *shape):
		"""Return a reshaped view of the tensor for forward passes."""
		if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
			shape = tuple(shape[0])
		out = Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad)
		out._prev = {self}

		def _backward() -> None:
			if self.requires_grad and out.grad is not None:
				self._add_grad(out.grad.reshape(self.data.shape))

		out._backward = _backward
		return out

	def transpose(self, *axes):
		"""Transpose the tensor and preserve the backward path."""
		if not axes:
			axes = tuple(reversed(range(self.data.ndim)))
		elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
			axes = tuple(axes[0])
		out = Tensor(np.transpose(self.data, axes=axes), requires_grad=self.requires_grad)
		out._prev = {self}
		inverse_axes = np.argsort(axes)

		def _backward() -> None:
			if self.requires_grad and out.grad is not None:
				self._add_grad(np.transpose(out.grad, axes=inverse_axes))

		out._backward = _backward
		return out

	@property
	def T(self):
		return self.transpose()

	def exp(self):
		"""Elementwise exponential."""
		out = Tensor(np.exp(self.data), requires_grad=self.requires_grad)
		out._prev = {self}

		def _backward() -> None:
			if self.requires_grad and out.grad is not None:
				self._add_grad(out.grad * out.data)

		out._backward = _backward
		return out

	def log(self):
		"""Elementwise natural logarithm."""
		out = Tensor(np.log(self.data), requires_grad=self.requires_grad)
		out._prev = {self}

		def _backward() -> None:
			if self.requires_grad and out.grad is not None:
				self._add_grad(out.grad / self.data)

		out._backward = _backward
		return out

	def tanh(self):
		"""Elementwise hyperbolic tangent."""
		out = Tensor(np.tanh(self.data), requires_grad=self.requires_grad)
		out._prev = {self}

		def _backward() -> None:
			if self.requires_grad and out.grad is not None:
				self._add_grad(out.grad * (1.0 - out.data ** 2))

		out._backward = _backward
		return out

	def sigmoid(self):
		"""Elementwise logistic sigmoid."""
		clipped = np.clip(self.data, -500, 500)
		data = 1.0 / (1.0 + np.exp(-clipped))
		out = Tensor(data, requires_grad=self.requires_grad)
		out._prev = {self}

		def _backward() -> None:
			if self.requires_grad and out.grad is not None:
				self._add_grad(out.grad * out.data * (1.0 - out.data))

		out._backward = _backward
		return out

	def relu(self):
		"""Elementwise rectified linear unit."""
		out = Tensor(np.maximum(0.0, self.data), requires_grad=self.requires_grad)
		out._prev = {self}

		def _backward() -> None:
			if self.requires_grad and out.grad is not None:
				self._add_grad(out.grad * (self.data > 0))

		out._backward = _backward
		return out

	def clip(self, min_value, max_value):
		"""Clamp tensor values into ``[min_value, max_value]``."""
		out = Tensor(np.clip(self.data, min_value, max_value), requires_grad=self.requires_grad)
		out._prev = {self}

		def _backward() -> None:
			if self.requires_grad and out.grad is not None:
				mask = (self.data >= min_value) & (self.data <= max_value)
				self._add_grad(out.grad * mask)

		out._backward = _backward
		return out

	def __array__(self, dtype=None):
		return np.asarray(self.data, dtype=dtype)

	def copy(self):
		return Tensor(self.data.copy(), requires_grad=self.requires_grad, name=self.name)

	@property
	def shape(self):
		return self.data.shape

	@property
	def ndim(self):
		return self.data.ndim

	def item(self):
		return self.data.item()

	def __len__(self):
		return len(self.data)

	def __repr__(self):
		return f"Tensor(data={self.data!r}, requires_grad={self.requires_grad}, grad={'set' if self.grad is not None else 'None'})"
