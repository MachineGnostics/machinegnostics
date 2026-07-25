"""Autograd-enabled tensor primitives for ANN models.

The implementation is intentionally minimal and NumPy-backed. It is designed to
support dense feed-forward networks first, while remaining easy to extend with
new tensor types later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union["Tensor", float, int, Sequence[float], np.ndarray]


def _ensure_array(data: ArrayLike) -> np.ndarray:
    if isinstance(data, Tensor):
        return data.data
    return np.asarray(data, dtype=np.float64)


def _unbroadcast(grad: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """Reduce a broadcast gradient back to the target shape."""

    grad = np.asarray(grad, dtype=np.float64)

    if target_shape == ():
        return np.asarray(grad).sum()

    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)

    for axis, size in enumerate(target_shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad.reshape(target_shape)


@dataclass
class TensorState:
    """Internal state container used by future extensions."""

    data: np.ndarray
    grad: Optional[np.ndarray] = None


class Tensor:
    """A NumPy-backed tensor with reverse-mode automatic differentiation."""

    def __init__(self, data: ArrayLike, _children=(), _op: str = "", requires_grad: bool = False):
        self.data = _ensure_array(data)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad}, op='{self._op}')"

    def numpy(self) -> np.ndarray:
        return np.asarray(self.data)

    def detach(self) -> "Tensor":
        return Tensor(self.data.copy(), requires_grad=False)

    def zero_grad(self) -> None:
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def _accumulate_grad(self, grad: np.ndarray) -> None:
        if self.requires_grad:
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            self.grad += grad

    def _as_tensor(self, other: ArrayLike) -> "Tensor":
        return other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)

    def __add__(self, other: ArrayLike) -> "Tensor":
        other = self._as_tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data + other.data, (self, other), "+", requires_grad=requires_grad)

        def _backward() -> None:
            if self.requires_grad:
                self._accumulate_grad(_unbroadcast(out.grad, self.data.shape))
            if other.requires_grad:
                other._accumulate_grad(_unbroadcast(out.grad, other.data.shape))

        out._backward = _backward
        return out

    def __radd__(self, other: ArrayLike) -> "Tensor":
        return self + other

    def __sub__(self, other: ArrayLike) -> "Tensor":
        return self + (-other)

    def __rsub__(self, other: ArrayLike) -> "Tensor":
        return other + (-self)

    def __mul__(self, other: ArrayLike) -> "Tensor":
        other = self._as_tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data * other.data, (self, other), "*", requires_grad=requires_grad)

        def _backward() -> None:
            if self.requires_grad:
                self._accumulate_grad(_unbroadcast(other.data * out.grad, self.data.shape))
            if other.requires_grad:
                other._accumulate_grad(_unbroadcast(self.data * out.grad, other.data.shape))

        out._backward = _backward
        return out

    def __rmul__(self, other: ArrayLike) -> "Tensor":
        return self * other

    def __truediv__(self, other: ArrayLike) -> "Tensor":
        other = self._as_tensor(other)
        return self * other ** -1

    def __rtruediv__(self, other: ArrayLike) -> "Tensor":
        other = self._as_tensor(other)
        return other / self

    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __pow__(self, power: float) -> "Tensor":
        out = Tensor(self.data ** power, (self,), f"**{power}", requires_grad=self.requires_grad)

        def _backward() -> None:
            if self.requires_grad:
                self._accumulate_grad((power * self.data ** (power - 1)) * out.grad)

        out._backward = _backward
        return out

    def matmul(self, other: ArrayLike) -> "Tensor":
        other = self._as_tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data @ other.data, (self, other), "matmul", requires_grad=requires_grad)

        def _backward() -> None:
            if self.requires_grad:
                self._accumulate_grad(out.grad @ other.data.T)
            if other.requires_grad:
                other._accumulate_grad(self.data.T @ out.grad)

        out._backward = _backward
        return out

    def __matmul__(self, other: ArrayLike) -> "Tensor":
        return self.matmul(other)

    def transpose(self, *axes: int) -> "Tensor":
        data = self.data.transpose(*axes) if axes else self.data.T
        out = Tensor(data, (self,), "transpose", requires_grad=self.requires_grad)

        def _backward() -> None:
            if self.requires_grad:
                if axes:
                    inverse = np.argsort(axes)
                    self._accumulate_grad(out.grad.transpose(*inverse))
                else:
                    self._accumulate_grad(out.grad.T)

        out._backward = _backward
        return out

    def reshape(self, *shape: int) -> "Tensor":
        out = Tensor(self.data.reshape(*shape), (self,), "reshape", requires_grad=self.requires_grad)

        def _backward() -> None:
            if self.requires_grad:
                self._accumulate_grad(out.grad.reshape(self.data.shape))

        out._backward = _backward
        return out

    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> "Tensor":
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum", requires_grad=self.requires_grad)

        def _backward() -> None:
            if not self.requires_grad:
                return
            grad = out.grad
            if axis is None:
                grad = np.ones_like(self.data) * grad
            else:
                expanded = grad
                if not keepdims:
                    axes = axis if isinstance(axis, tuple) else (axis,)
                    for ax in sorted(axes):
                        expanded = np.expand_dims(expanded, axis=ax)
                grad = np.ones_like(self.data) * expanded
            self._accumulate_grad(grad)

        out._backward = _backward
        return out

    def max(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> "Tensor":
        out = Tensor(self.data.max(axis=axis, keepdims=keepdims), (self,), "max", requires_grad=self.requires_grad)

        def _backward() -> None:
            if not self.requires_grad:
                return
            grad = out.grad
            if axis is None:
                mask = self.data == self.data.max()
                count = mask.sum()
                self._accumulate_grad(mask.astype(np.float64) * grad / max(count, 1))
                return

            if not keepdims:
                axes = axis if isinstance(axis, tuple) else (axis,)
                for ax in sorted(axes):
                    grad = np.expand_dims(grad, axis=ax)

            max_values = self.data.max(axis=axis, keepdims=True)
            mask = self.data == max_values
            counts = mask.sum(axis=axis, keepdims=True)
            self._accumulate_grad(mask.astype(np.float64) * grad / np.maximum(counts, 1))

        out._backward = _backward
        return out

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> "Tensor":
        if axis is None:
            divisor = self.data.size
        elif isinstance(axis, tuple):
            divisor = np.prod([self.data.shape[index] for index in axis])
        else:
            divisor = self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) / divisor

    def abs(self) -> "Tensor":
        out = Tensor(np.abs(self.data), (self,), "abs", requires_grad=self.requires_grad)

        def _backward() -> None:
            if self.requires_grad:
                self._accumulate_grad(np.sign(self.data) * out.grad)

        out._backward = _backward
        return out

    def exp(self) -> "Tensor":
        out = Tensor(np.exp(self.data), (self,), "exp", requires_grad=self.requires_grad)

        def _backward() -> None:
            if self.requires_grad:
                self._accumulate_grad(out.data * out.grad)

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        out = Tensor(np.log(self.data), (self,), "log", requires_grad=self.requires_grad)

        def _backward() -> None:
            if self.requires_grad:
                self._accumulate_grad(out.grad / self.data)

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        out = Tensor(np.tanh(self.data), (self,), "tanh", requires_grad=self.requires_grad)

        def _backward() -> None:
            if self.requires_grad:
                self._accumulate_grad((1.0 - out.data ** 2) * out.grad)

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        data = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(data, (self,), "sigmoid", requires_grad=self.requires_grad)

        def _backward() -> None:
            if self.requires_grad:
                self._accumulate_grad(out.data * (1.0 - out.data) * out.grad)

        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(0.0, self.data), (self,), "relu", requires_grad=self.requires_grad)

        def _backward() -> None:
            if self.requires_grad:
                self._accumulate_grad((self.data > 0.0).astype(np.float64) * out.grad)

        out._backward = _backward
        return out

    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        """Backpropagate from the current tensor.

        If the tensor is non-scalar, a matching gradient must be provided.
        """

        topo = []
        visited = set()

        def build_topo(tensor: "Tensor") -> None:
            if tensor not in visited:
                visited.add(tensor)
                for parent in tensor._prev:
                    build_topo(parent)
                topo.append(tensor)

        build_topo(self)

        if grad is None:
            if self.data.size != 1:
                raise ValueError("backward() requires an explicit gradient for non-scalar tensors.")
            grad = np.ones_like(self.data)

        self.grad = np.asarray(grad, dtype=np.float64)
        for tensor in reversed(topo):
            tensor._backward()
