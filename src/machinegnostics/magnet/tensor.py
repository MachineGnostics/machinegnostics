"""Autograd-enabled tensor primitives for ANN models.

The implementation is intentionally minimal and NumPy-backed. It is designed to
support dense feed-forward networks first, while remaining easy to extend with
new tensor types later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import logging

import numpy as np

from machinegnostics.magcal import DataConversion
from machinegnostics.magcal.util.logging import get_logger

from .engine.gnostic_engine import GnosticEngine

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


@dataclass
class GnosticState:
    """Stores the latest gnostic quantities computed for a tensor."""

    weights: Optional[np.ndarray] = None
    error: Optional[np.ndarray] = None
    gnostic_error: Optional[np.ndarray] = None
    activation: Optional[np.ndarray] = None
    fi: Optional[np.ndarray] = None
    fj: Optional[np.ndarray] = None
    hi: Optional[np.ndarray] = None
    hj: Optional[np.ndarray] = None
    rentropy: Optional[np.ndarray] = None
    ientropy: Optional[np.ndarray] = None
    jentropy: Optional[np.ndarray] = None


def _safe_reference(reference: float | None, values: np.ndarray) -> float:
    if reference is None:
        reference = float(np.median(values))
    reference = float(reference)
    if not np.isfinite(reference) or abs(reference) < 1e-12:
        reference = 1e-12 if reference >= 0 else -1e-12
    return reference


def _compute_gnostic_bundle(
    values: ArrayLike,
    *,
    scale_param: str | float = "auto",
    activation_type: str = "fi",
    reference: float | None = None,
    verbose: bool = False,
) -> dict[str, np.ndarray]:
    """Compute gnostic weights, characteristics, entropy, and activation for values."""

    arr = _ensure_array(values)
    if arr.size == 0:
        raise ValueError("Gnostic computations require non-empty data.")

    z0 = _safe_reference(reference, arr)
    converted = DataConversion._convert_az(arr - z0)

    engine = GnosticEngine(verbose=verbose)
    weights = engine._get_gnostic_weights(converted, scale_param=scale_param)
    activation = engine._get_activation(converted, scale_param=scale_param, activation_type=activation_type)

    fi = engine._get_fi()
    fj = engine._get_fj()
    hi = engine._get_hi()
    hj = engine._get_hj()
    rentropy = engine._get_re()
    ientropy = 1.0 - fi
    jentropy = fj - 1.0

    return {
        "reference": np.asarray(z0, dtype=np.float64),
        "converted": np.asarray(converted, dtype=np.float64),
        "weights": np.asarray(weights, dtype=np.float64),
        "activation": np.asarray(activation, dtype=np.float64),
        "fi": np.asarray(fi, dtype=np.float64),
        "fj": np.asarray(fj, dtype=np.float64),
        "hi": np.asarray(hi, dtype=np.float64),
        "hj": np.asarray(hj, dtype=np.float64),
        "rentropy": np.asarray(rentropy, dtype=np.float64),
        "ientropy": np.asarray(ientropy, dtype=np.float64),
        "jentropy": np.asarray(jentropy, dtype=np.float64),
    }


class Tensor:
    """A NumPy-backed tensor with reverse-mode automatic differentiation."""

    def __init__(self, data: ArrayLike, _children=(), _op: str = "", requires_grad: bool = False):
        self.data = _ensure_array(data)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.logger = get_logger(self.__class__.__name__, logging.WARNING)
        self.gnostic = GnosticState()
        self.gnostic_weights = None
        self.gnostic_error = None
        self.gnostic_activation = None
        self.gnostic_characteristics = None
        self.rentropy = None
        self.ientropy = None
        self.jentropy = None
        self.fi = None
        self.fj = None
        self.hi = None
        self.hj = None

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

    def compute_gnostic_weights(self, scale_param: str | float = "auto", reference: float | None = None) -> np.ndarray:
        """Compute gnostic weights for the tensor values."""

        self.logger.info("Computing gnostic weights for tensor values.")
        bundle = _compute_gnostic_bundle(self.data, scale_param=scale_param, reference=reference)
        self._cache_gnostic_bundle(bundle)
        return bundle["weights"]

    def compute_gnostic_error(
        self,
        target: ArrayLike,
        *,
        y_pred: ArrayLike | None = None,
        scale_param: str | float = "auto",
        reference: float | None = None,
        use_gnostic_weights: bool = True,
    ) -> np.ndarray:
        """Compute normal error and its gnostic-weighted version."""

        self.logger.info("Computing gnostic error.")
        prediction = self.data if y_pred is None else _ensure_array(y_pred)
        target_array = _ensure_array(target)
        normal_error = prediction - target_array
        bundle = _compute_gnostic_bundle(normal_error, scale_param=scale_param, reference=reference)
        gnostic_error = normal_error * bundle["weights"] if use_gnostic_weights else normal_error

        bundle["error"] = normal_error
        bundle["gnostic_error"] = gnostic_error
        self._cache_gnostic_bundle(bundle)
        return gnostic_error

    def compute_gnostic_characteristics(
        self,
        data: ArrayLike | None = None,
        *,
        scale_param: str | float = "auto",
        reference: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute the gnostic characteristics bundle for the tensor or provided data."""

        self.logger.info("Computing gnostic characteristics.")
        values = self.data if data is None else data
        bundle = _compute_gnostic_bundle(values, scale_param=scale_param, reference=reference)
        self._cache_gnostic_bundle(bundle)
        return bundle

    def compute_gnostic_activation(
        self,
        activation_type: str = "fi",
        *,
        scale_param: str | float = "auto",
        reference: float | None = None,
    ) -> np.ndarray:
        """Compute a gnostic activation for the tensor values."""

        self.logger.info("Computing gnostic activation.")
        bundle = _compute_gnostic_bundle(self.data, scale_param=scale_param, activation_type=activation_type, reference=reference)
        self._cache_gnostic_bundle(bundle)
        return bundle["activation"]

    def compute_gnostic_characteristic_loss(
        self,
        data: ArrayLike | None = None,
        *,
        scale_param: str | float = "auto",
        reference: float | None = None,
        mode: str = "mean",
    ) -> float:
        """Compute the secondary gnostic characteristic loss."""

        bundle = self.compute_gnostic_characteristics(data=data, scale_param=scale_param, reference=reference)
        means = {
            "fi": float(np.mean(bundle["fi"])),
            "fj": float(np.mean(bundle["fj"])),
            "hi": float(np.mean(bundle["hi"])),
            "hj": float(np.mean(bundle["hj"])),
        }

        if mode == "mean":
            return float(np.mean(list(means.values())))
        if mode not in means:
            raise ValueError("mode must be one of ['mean', 'fi', 'fj', 'hi', 'hj']")
        return means[mode]

    def _cache_gnostic_bundle(self, bundle: dict[str, np.ndarray]) -> None:
        self.gnostic = GnosticState(
            weights=bundle.get("weights"),
            error=bundle.get("error"),
            gnostic_error=bundle.get("gnostic_error"),
            activation=bundle.get("activation"),
            fi=bundle.get("fi"),
            fj=bundle.get("fj"),
            hi=bundle.get("hi"),
            hj=bundle.get("hj"),
            rentropy=bundle.get("rentropy"),
            ientropy=bundle.get("ientropy"),
            jentropy=bundle.get("jentropy"),
        )
        self.gnostic_weights = bundle.get("weights")
        self.gnostic_error = bundle.get("gnostic_error")
        self.gnostic_activation = bundle.get("activation")
        self.gnostic_characteristics = bundle
        self.rentropy = bundle.get("rentropy")
        self.ientropy = bundle.get("ientropy")
        self.jentropy = bundle.get("jentropy")
        self.fi = bundle.get("fi")
        self.fj = bundle.get("fj")
        self.hi = bundle.get("hi")
        self.hj = bundle.get("hj")

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
