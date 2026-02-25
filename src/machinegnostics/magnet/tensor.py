'''
Tensor class for autograd in Magnet. This is a minimal implementation to support the core operations needed for Dense layers and activations.

Author: Nirmal Parmar
'''

import numpy as np

class Tensor:
    def __init__(self, data, name=None, requires_grad=True):
        self.data = np.array(data, dtype=np.float64)
        self.name = name
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data)
        
        # Computational Graph references
        self._prev = []
        self._backward_fn = lambda: None

    def __repr__(self):
        return f"<Tensor(shape={self.data.shape}, name={self.name})>"

    @property
    def shape(self):
        return self.data.shape

    # --- Addition (with Broadcasting support) ---
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data + other.data)
        out._prev = [self, other]

        def _backward():
            # If self.data was (10, 5) and other.data was (5,), 
            # NumPy broadcasted. We must sum gradients back to original shape.
            if self.requires_grad:
                self.grad += self._match_shape(out.grad, self.shape)
            if other.requires_grad:
                other.grad += self._match_shape(out.grad, other.shape)
        
        out._backward_fn = _backward
        return out

    # --- Matrix Multiplication (The core of Dense layers) ---
    def __matmul__(self, other):
        # other is usually the Weights matrix
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.matmul(self.data, other.data))
        out._prev = [self, other]

        def _backward():
            if self.requires_grad:
                # dL/dX = dL/dOut @ W.T
                self.grad += np.matmul(out.grad, other.data.T)
            if other.requires_grad:
                # dL/dW = X.T @ dL/dOut
                other.grad += np.matmul(self.data.T, out.grad)
        
        out._backward_fn = _backward
        return out

    # --- Utility for Gradient Broadcasting ---
    def _match_shape(self, grad, target_shape):
        """Sums gradients across broadcasted dimensions to match the target_shape."""
        while grad.ndim > len(target_shape):
            grad = grad.sum(axis=0)
        for axis, size in enumerate(target_shape):
            if size == 1:
                grad = grad.sum(axis=axis, keepdims=True)
        return grad

    # --- The Engine ---
    def backward(self, grad=None):
        if grad is None:
            # If this is the loss tensor, start with 1.0
            grad = np.ones_like(self.data)
        self.grad = grad

        # Build topological sort to ensure correct execution order
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build_topo(parent)
                topo.append(v)
        build_topo(self)

        for node in reversed(topo):
            node._backward_fn()

    # Python dunder methods for convenience
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data * other.data, _children=(self, other))
        def _backward():
            if self.requires_grad: self.grad += other.data * out.grad
            if other.requires_grad: other.grad += self.data * out.grad
        out._backward_fn = _backward
        return out