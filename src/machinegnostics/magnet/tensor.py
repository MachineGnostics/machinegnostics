"""Tensor and autograd primitives for Magnet.

Author: Nirmal Parmar

Developer Notes:
- This module provides a NumPy-backed tensor with reverse-mode autodiff.
- Broadcasting-aware gradients are handled internally for elementwise ops.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np



class Tensor:
    """
    Magnet Tensor: NumPy-backed tensor with reverse-mode autodiff.

    ---
    User Documentation
    ------------------
    The `Tensor` class is the core data structure for all neural network computations in Magnet.
    It wraps a NumPy array and supports automatic differentiation (autograd) for building and training neural networks.

    **Key Features:**
    - Drop-in replacement for NumPy arrays in most mathematical operations.
    - Tracks computation graph for gradients (reverse-mode autodiff).
    - Supports broadcasting, elementwise ops, matrix multiplication, reductions, and more.
    - Gradients are accumulated in `.grad` if `requires_grad=True`.
    - Compatible with all Magnet layers, losses, and optimizers.

    **Basic Usage:**
    ```python
    from machinegnostics.magnet import Tensor
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x * 2 + 1
    z = y.sum()
    z.backward()  # Computes dz/dx for all x
    print(x.grad) # Shows gradients
    ```

    **Parameters**
    -------------
    data : array-like
        Input data (NumPy array, list, etc.).
    name : str, optional
        Optional name for debugging/tracing.
    requires_grad : bool, default=True
        If True, gradients are tracked and accumulated in `.grad`.
    device : str, default="cpu"
        Device marker (currently only "cpu" supported).
    dtype : np.dtype, default=np.float64
        Data type for storage and computation.

    **Attributes**
    -------------
    .data : np.ndarray
        The underlying NumPy array.
    .grad : np.ndarray or None
        Accumulated gradients (if requires_grad=True).
    .shape : tuple
        Shape of the tensor.
    .requires_grad : bool
        Whether this tensor tracks gradients.
    .device : str
        Device marker ("cpu").
    .name : str or None
        Optional name for debugging.

    ---
    Developer Notes
    ---------------
    - The Tensor class implements reverse-mode autodiff using a computation graph.
    - Each operation creates a new Tensor and stores references to its parents in `_prev`.
    - The `_backward_fn` closure for each Tensor encodes the local gradient rule for that operation.
    - Calling `.backward()` on a scalar Tensor traverses the graph in topological order and accumulates gradients in `.grad` for all ancestors with `requires_grad=True`.
    - Broadcasting and shape reduction for gradients are handled in `_reduce_grad_to_shape`.
    - All mathematical operations (add, mul, matmul, pow, exp, log, etc.) are overloaded to support both Tensor and scalar/array inputs.
    - The implementation is NumPy-only and CPU-only for simplicity and transparency.
    - Designed for extensibility: new ops can be added by following the pattern of creating a new Tensor, storing parents, and defining a `_backward_fn`.
    - For efficiency, gradients are only allocated and accumulated if `requires_grad=True`.
    - This class is the foundation for all Magnet neural network layers, losses, and optimizers.
    """

    def __init__(
        self,
        data,
        name: Optional[str] = None,
        requires_grad: bool = True,
        device: str = "cpu",
        dtype=np.float64,
    ):
        self.data = np.array(data, dtype=dtype)
        self.name = name
        self.requires_grad = bool(requires_grad)
        self.device = device
        self.grad = np.zeros_like(self.data, dtype=self.data.dtype) if self.requires_grad else None
        self._prev: set[Tensor] = set()
        self._backward_fn: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        return (
            f"Tensor(shape={self.data.shape}, dtype={self.data.dtype}, "
            f"requires_grad={self.requires_grad}, device='{self.device}', name={self.name})"
        )

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @staticmethod
    def _ensure_tensor(other) -> "Tensor":
        if isinstance(other, Tensor):
            return other
        return Tensor(other, requires_grad=False)

    @staticmethod
    def _reduce_grad_to_shape(grad: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
        while grad.ndim > len(target_shape):
            grad = grad.sum(axis=0)
        for axis, size in enumerate(target_shape):
            if size == 1 and grad.shape[axis] != 1:
                grad = grad.sum(axis=axis, keepdims=True)
        return grad

    def zero_grad(self) -> None:
        if self.requires_grad:
            self.grad = np.zeros_like(self.data, dtype=self.data.dtype)

    def detach(self) -> "Tensor":
        return Tensor(self.data.copy(), name=self.name, requires_grad=False, device=self.device, dtype=self.data.dtype)

    def item(self):
        return self.data.item()

    def numpy(self) -> np.ndarray:
        return self.data

    def __add__(self, other):
        other = self._ensure_tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            device=self.device,
            dtype=self.data.dtype,
        )
        out._prev = {self, other}

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += self._reduce_grad_to_shape(out.grad, self.shape)
            if other.requires_grad:
                other.grad += self._reduce_grad_to_shape(out.grad, other.shape)

        out._backward_fn = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = self._ensure_tensor(other)
        return self.__add__(-other)

    def __rsub__(self, other):
        other = self._ensure_tensor(other)
        return other.__sub__(self)

    def __neg__(self):
        out = Tensor(-self.data, requires_grad=self.requires_grad, device=self.device, dtype=self.data.dtype)
        out._prev = {self}

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad -= out.grad

        out._backward_fn = _backward
        return out

    def __mul__(self, other):
        other = self._ensure_tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            device=self.device,
            dtype=self.data.dtype,
        )
        out._prev = {self, other}

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += self._reduce_grad_to_shape(out.grad * other.data, self.shape)
            if other.requires_grad:
                other.grad += self._reduce_grad_to_shape(out.grad * self.data, other.shape)

        out._backward_fn = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = self._ensure_tensor(other)
        return self * other.pow(-1.0)

    def __rtruediv__(self, other):
        other = self._ensure_tensor(other)
        return other.__truediv__(self)

    def pow(self, exponent: float):
        out_data = self.data ** exponent
        out = Tensor(out_data, requires_grad=self.requires_grad, device=self.device, dtype=self.data.dtype)
        out._prev = {self}

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += out.grad * exponent * (self.data ** (exponent - 1.0))

        out._backward_fn = _backward
        return out

    def sqrt(self):
        return self.pow(0.5)

    def __pow__(self, exponent: float):
        return self.pow(exponent)

    def exp(self):
        out_data = np.exp(self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad, device=self.device, dtype=self.data.dtype)
        out._prev = {self}

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += out.grad * out_data

        out._backward_fn = _backward
        return out

    def log(self):
        out_data = np.log(self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad, device=self.device, dtype=self.data.dtype)
        out._prev = {self}

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += out.grad / self.data

        out._backward_fn = _backward
        return out

    def matmul(self, other):
        other = self._ensure_tensor(other)
        out = Tensor(
            np.matmul(self.data, other.data),
            requires_grad=self.requires_grad or other.requires_grad,
            device=self.device,
            dtype=self.data.dtype,
        )
        out._prev = {self, other}

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += np.matmul(out.grad, np.swapaxes(other.data, -1, -2))
            if other.requires_grad:
                other.grad += np.matmul(np.swapaxes(self.data, -1, -2), out.grad)

        out._backward_fn = _backward
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    def sum(self, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False):
        out_data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad, device=self.device, dtype=self.data.dtype)
        out._prev = {self}

        if axis is None:
            axis_tuple: Tuple[int, ...] = tuple(range(self.data.ndim))
        elif isinstance(axis, tuple):
            axis_tuple = axis
        elif isinstance(axis, list):
            axis_tuple = tuple(axis)
        else:
            axis_tuple = (axis,)

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                grad = out.grad
                if not keepdims and axis is not None:
                    for ax in sorted(axis_tuple):
                        grad = np.expand_dims(grad, axis=ax)
                self.grad += np.ones_like(self.data) * grad

        out._backward_fn = _backward
        return out

    def mean(self, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False):
        if axis is None:
            denom = self.data.size
        elif isinstance(axis, (tuple, list)):
            denom = int(np.prod([self.data.shape[ax] for ax in axis]))
        else:
            denom = self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) / denom

    def reshape(self, *shape: int):
        out = Tensor(
            self.data.reshape(*shape),
            requires_grad=self.requires_grad,
            device=self.device,
            dtype=self.data.dtype,
        )
        out._prev = {self}

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += out.grad.reshape(self.shape)

        out._backward_fn = _backward
        return out

    @property
    def T(self):
        out = Tensor(
            self.data.T,
            requires_grad=self.requires_grad,
            device=self.device,
            dtype=self.data.dtype,
        )
        out._prev = {self}

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += out.grad.T

        out._backward_fn = _backward
        return out

    def clip(self, min_value: float, max_value: float):
        out_data = np.clip(self.data, min_value, max_value)
        out = Tensor(out_data, requires_grad=self.requires_grad, device=self.device, dtype=self.data.dtype)
        out._prev = {self}

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                mask = (self.data >= min_value) & (self.data <= max_value)
                self.grad += out.grad * mask

        out._backward_fn = _backward
        return out

    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        """Backpropagate from current node to all graph ancestors."""
        if grad is None:
            if self.data.size != 1:
                raise ValueError("grad must be provided for non-scalar Tensor backward().")
            grad = np.ones_like(self.data)

        topo: list[Tensor] = []
        visited: set[int] = set()

        def build_topo(node: Tensor):
            node_id = id(node)
            if node_id in visited:
                return
            visited.add(node_id)
            for parent in node._prev:
                build_topo(parent)
            topo.append(node)

        build_topo(self)

        for node in topo:
            if node.requires_grad and node.grad is None:
                node.grad = np.zeros_like(node.data, dtype=node.data.dtype)

        self.grad = self.grad + grad if self.grad is not None else grad

        for node in reversed(topo):
            node._backward_fn()