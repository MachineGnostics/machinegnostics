"""Torch-backed tensor facade for MAGNET.

Developer note
-------------
Author: Nirmal Parmar, Machine Gnostics

This class is the only tensor object that MAGNET users should need. It hides
the torch implementation detail behind the existing MAGNET API, keeps NumPy
style inspection helpers such as ``data`` and ``grad``, and lets the rest of
the library keep its current model / layer / loss design.

Bird's-eye view
---------------
- public construction still looks like a MAGNET tensor;
- arithmetic and autograd run on torch internally;
- ``data`` and ``grad`` remain NumPy-friendly for the user boundary;
- device selection comes from ``machinegnostics.magnet.configure``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..utils.logging import get_logger
from .runtime import friendly_device_name, get_torch_device, get_torch_dtype, to_numpy, to_torch

logger = get_logger(__name__)


def _normalize_axes(axes: tuple[Any, ...]) -> tuple[int, ...]:
	if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
		axes = tuple(axes[0])
	return tuple(int(axis) for axis in axes)


def unbroadcast(gradient: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
	"""Reduce a broadcasted gradient back to the original tensor shape."""
	gradient = np.asarray(gradient)
	if gradient.shape == shape:
		return gradient
	while gradient.ndim > len(shape):
		gradient = gradient.sum(axis=0)
	for axis, size in enumerate(shape):
		if size == 1 and gradient.shape[axis] != 1:
			gradient = gradient.sum(axis=axis, keepdims=True)
	return gradient.reshape(shape)


class Tensor:
	"""A MAGNET tensor with hidden torch autograd support."""

	__array_priority__ = 1000

	def __init__(self, data, requires_grad: bool = False, name: str | None = None, device: str | None = None, dtype: str | None = None):
		self.name = name
		self._tensor = to_torch(data, requires_grad=requires_grad, device=device, dtype=dtype)
		logger.debug("Tensor initialized with shape %s on %s.", tuple(self.shape), self.device)

	@classmethod
	def from_torch(cls, tensor: torch.Tensor, name: str | None = None) -> "Tensor":
		obj = cls.__new__(cls)
		obj.name = name
		obj._tensor = tensor
		return obj

	@staticmethod
	def _ensure_tensor(value, device: str | None = None, dtype: str | None = None) -> "Tensor":
		if isinstance(value, Tensor):
			return value
		return Tensor(value, device=device, dtype=dtype)

	@property
	def data(self):
		return to_numpy(self._tensor)

	@data.setter
	def data(self, value) -> None:
		self._tensor = to_torch(value, requires_grad=self.requires_grad, device=str(self._tensor.device), dtype=str(self.dtype).split(".")[-1])

	@property
	def grad(self):
		if self._tensor.grad is None:
			return None
		return to_numpy(self._tensor.grad)

	@grad.setter
	def grad(self, value) -> None:
		if value is None:
			self._tensor.grad = None
			return
		self._tensor.grad = to_torch(value, device=str(self._tensor.device), dtype=str(self.dtype).split(".")[-1])

	@property
	def requires_grad(self) -> bool:
		return bool(self._tensor.requires_grad)

	@requires_grad.setter
	def requires_grad(self, value: bool) -> None:
		self._tensor.requires_grad_(bool(value))

	@property
	def shape(self) -> tuple[int, ...]:
		return tuple(self._tensor.shape)

	@property
	def ndim(self) -> int:
		return self._tensor.ndim

	@property
	def size(self) -> int:
		return int(self._tensor.numel())

	@property
	def device(self) -> str:
		return friendly_device_name(self._tensor.device)

	@property
	def dtype(self):
		return self._tensor.dtype

	def _add_grad(self, gradient) -> None:
		if not self.requires_grad:
			return
		grad_tensor = to_torch(gradient, device=self.device, dtype=str(self.dtype).split(".")[-1])
		if self._tensor.grad is None:
			self._tensor.grad = grad_tensor.clone()
		else:
			self._tensor.grad = self._tensor.grad + grad_tensor

	def zero_grad(self) -> None:
		if self.requires_grad:
			self._tensor.grad = torch.zeros_like(self._tensor)

	def detach(self) -> "Tensor":
		return Tensor.from_torch(self._tensor.detach().clone(), name=self.name)

	def clone(self) -> "Tensor":
		return Tensor.from_torch(self._tensor.clone(), name=self.name)

	def numpy(self) -> np.ndarray:
		return self.data

	def item(self):
		return self._tensor.item()

	def to(self, device: str | None = None, dtype: str | None = None) -> "Tensor":
		return Tensor.from_torch(self._tensor.to(device=get_torch_device(device), dtype=get_torch_dtype(dtype)), name=self.name)

	def backward(self, gradient=None) -> None:
		if not self.requires_grad:
			return
		if gradient is None:
			if self._tensor.numel() != 1:
				raise ValueError("gradient must be provided for non-scalar tensors")
			self._tensor.backward()
			return
		self._tensor.backward(to_torch(gradient, device=self.device, dtype=str(self.dtype).split(".")[-1]))

	def __array__(self, dtype=None):
		array = self.data
		return array.astype(dtype) if dtype is not None else array

	def __len__(self):
		return len(self.data)

	def __float__(self):
		return float(self.item())

	def __repr__(self):
		return f"Tensor(shape={self.shape}, device={self.device}, requires_grad={self.requires_grad})"

	def _binary_op(self, other, op):
		other = self._ensure_tensor(other, device=self.device, dtype=str(self.dtype).split(".")[-1])
		return Tensor.from_torch(op(self._tensor, other._tensor))

	def __add__(self, other):
		return self._binary_op(other, torch.add)

	def __radd__(self, other):
		return self.__add__(other)

	def __sub__(self, other):
		return self._binary_op(other, torch.sub)

	def __rsub__(self, other):
		other = self._ensure_tensor(other, device=self.device, dtype=str(self.dtype).split(".")[-1])
		return Tensor.from_torch(torch.sub(other._tensor, self._tensor))

	def __mul__(self, other):
		return self._binary_op(other, torch.mul)

	def __rmul__(self, other):
		return self.__mul__(other)

	def __truediv__(self, other):
		return self._binary_op(other, torch.div)

	def __rtruediv__(self, other):
		other = self._ensure_tensor(other, device=self.device, dtype=str(self.dtype).split(".")[-1])
		return Tensor.from_torch(torch.div(other._tensor, self._tensor))

	def __pow__(self, power):
		if isinstance(power, Tensor):
			return Tensor.from_torch(self._tensor ** power._tensor)
		return Tensor.from_torch(self._tensor ** power)

	def __neg__(self):
		return Tensor.from_torch(-self._tensor)

	def __matmul__(self, other):
		other = self._ensure_tensor(other, device=self.device, dtype=str(self.dtype).split(".")[-1])
		return Tensor.from_torch(self._tensor @ other._tensor)

	def __getitem__(self, item):
		return Tensor.from_torch(self._tensor.__getitem__(item))

	def sum(self, axis=None, keepdims: bool = False):
		if axis is None:
			result = self._tensor.sum()
		else:
			axes = _normalize_axes(axis if isinstance(axis, tuple) else (axis,))
			result = self._tensor.sum(dim=axes, keepdim=keepdims)
		return Tensor.from_torch(result)

	def mean(self, axis=None, keepdims: bool = False):
		if axis is None:
			result = self._tensor.mean()
		else:
			axes = _normalize_axes(axis if isinstance(axis, tuple) else (axis,))
			result = self._tensor.mean(dim=axes, keepdim=keepdims)
		return Tensor.from_torch(result)

	def reshape(self, *shape):
		if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
			shape = tuple(shape[0])
		return Tensor.from_torch(self._tensor.reshape(*shape))

	def transpose(self, *axes):
		if not axes:
			axes = tuple(reversed(range(self._tensor.ndim)))
		axes = _normalize_axes(axes)
		return Tensor.from_torch(self._tensor.permute(*axes))

	@property
	def T(self):
		return self.transpose()

	def exp(self):
		return Tensor.from_torch(torch.exp(self._tensor))

	def log(self):
		return Tensor.from_torch(torch.log(self._tensor))

	def tanh(self):
		return Tensor.from_torch(torch.tanh(self._tensor))

	def sigmoid(self):
		return Tensor.from_torch(torch.sigmoid(self._tensor))

	def relu(self):
		return Tensor.from_torch(torch.relu(self._tensor))

	def clip(self, min_value, max_value):
		return Tensor.from_torch(torch.clamp(self._tensor, min=min_value, max=max_value))

	def sqrt(self):
		return Tensor.from_torch(torch.sqrt(self._tensor))

	def abs(self):
		return Tensor.from_torch(torch.abs(self._tensor))

	def copy(self):
		return Tensor.from_torch(self._tensor.detach().clone(), name=self.name)
