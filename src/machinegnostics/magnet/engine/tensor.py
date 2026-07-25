import numpy as np

def _unbroadcast(grad, target_shape):
    """Sum gradient axes introduced by broadcasting."""
    grad = np.asarray(grad, dtype=np.float64)

    if target_shape == ():
        return np.asarray(grad).sum()

    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)

    for axis, size in enumerate(target_shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad.reshape(target_shape)

# def _unbroadcast(grad, shape):
#     """Sum-reduce a gradient back down to the original (pre-broadcast) shape."""
#     while grad.ndim > len(shape):
#         grad= grad.sum(axis=0)
#     for i, dim in enumerate(shape):
#         if dim == 1:
#             grad= grad.sum(axis=i, keepdims=True)
#     return grad
    

class Tensor:
    """
    A NumPy array that records the operations used to create it,
    enabling automatic reverse-mode differentiation via .backward().
    """

    def __init__(self, data, _children=(), _op="", requires_grad=True):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(shape={self.shape}, op='{self._op}')"

    def _accumulate_grad(self, grad):
        if self.requires_grad:
            self.grad += grad

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data + other.data, (self, other), "+", requires_grad=requires_grad)

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(_unbroadcast(out.grad, self.data.shape))
            if other.requires_grad:
                other._accumulate_grad(_unbroadcast(out.grad, other.data.shape))

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data * other.data, (self, other), "*", requires_grad=requires_grad)

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(_unbroadcast(other.data * out.grad, self.data.shape))
            if other.requires_grad:
                other._accumulate_grad(_unbroadcast(self.data * out.grad, other.data.shape))

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __pow__(self, power):
        out = Tensor(self.data ** power, (self,), f"**{power}", requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self._accumulate_grad((power * self.data ** (power - 1)) * out.grad)

        out._backward = _backward
        return out

    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data @ other.data, (self, other), "matmul", requires_grad=requires_grad)

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(out.grad @ other.data.T)
            if other.requires_grad:
                other._accumulate_grad(self.data.T @ out.grad)

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), "relu", requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self._accumulate_grad((self.data > 0) * out.grad)

        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(self.data.sum(), (self,), "sum", requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(np.ones_like(self.data) * out.grad)

        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()

        def build_topo(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for parent in tensor._prev:
                    build_topo(parent)
                topo.append(tensor)

        build_topo(self)
        self.grad = np.ones_like(self.data)
        for tensor in reversed(topo):
            tensor._backward()