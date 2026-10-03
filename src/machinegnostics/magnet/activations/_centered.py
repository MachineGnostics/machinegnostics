"""Shared helpers for centered gnostic activations."""

from __future__ import annotations

import numpy as np
import torch

from ..core.tensor import Tensor
from .base import Activation

EPS = 1e-6
MAX_ABS_TWO_THETA = 30.0
MAX_MAGNITUDE = 1e6


def _safe_scale_value(S: float) -> float:
    scale = abs(float(S))
    if scale < EPS:
        scale = EPS
    return scale


def _initial_scale_parameter(initial_S: float, learnable_S: bool) -> np.ndarray:
    scale = _safe_scale_value(initial_S)
    if not learnable_S:
        return np.array([scale], dtype=np.float64)

    bounded = min(max(scale / 2.0, EPS), 1.0 - EPS)
    raw = np.log(bounded) - np.log1p(-bounded)
    return np.array([raw], dtype=np.float64)


def _effective_scale(scale_parameter: torch.Tensor, learnable_S: bool) -> torch.Tensor:
    if learnable_S:
        scale = 2.0 * torch.sigmoid(scale_parameter)
        return torch.clamp(scale, min=EPS, max=2.0 - EPS)
    return torch.clamp(torch.abs(scale_parameter), min=EPS)


def _theta_components(
    x_tensor: torch.Tensor,
    scale_parameter: torch.Tensor,
    z0_tensor: torch.Tensor,
    learnable_S: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = _effective_scale(scale_parameter, learnable_S)
    theta = (x_tensor - z0_tensor) / scale
    two_theta_raw = 2.0 * theta
    two_theta = torch.clamp(two_theta_raw, min=-MAX_ABS_TWO_THETA, max=MAX_ABS_TWO_THETA)
    active = (torch.abs(two_theta_raw) < MAX_ABS_TWO_THETA).to(dtype=x_tensor.dtype)
    return scale, theta, two_theta, active


def _scalar_parameter_gradient(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return value.sum().reshape(reference.shape)


class CenteredGnosticActivation(Activation):
    """Activation with learnable center ``z0`` and positive scale ``S``.

    Learnable scales are stored as an unconstrained raw parameter and mapped to
    the effective positive scale used in the forward pass via
    ``S = 2σ(S_raw)``. Fixed scales bypass the sigmoid gate and use their value
    directly.

    When ``initial_z0=None``, the concept center is initialized to the median
    of the input data on the first forward pass.
    """

    def __init__(
        self,
        *,
        learnable_S: bool = True,
        learnable_z0: bool = True,
        initial_S: float = 1.0,
        initial_z0: float | None = None,
        name: str | None = None,
        verbose: bool = False,
    ):
        super().__init__(name=name, verbose=verbose)
        if initial_S <= 0:
            raise ValueError(f"initial_S must be positive, got {initial_S}")

        self.learnable_S = learnable_S
        self.learnable_z0 = learnable_z0
        self.verbose = verbose
        self._initial_z0_user = initial_z0
        self._z0_initialized = initial_z0 is not None

        # Initialize z0 with user value or placeholder (will be computed from data)
        z0_value = initial_z0 if initial_z0 is not None else 0.0
        self.S = Tensor(
            _initial_scale_parameter(initial_S, learnable_S),
            requires_grad=learnable_S,
            name=f"{self.name}_S" if self.name else f"{self.__class__.__name__}_S",
        )
        self.z0 = Tensor(
            np.array([z0_value], dtype=np.float64),
            requires_grad=learnable_z0,
            name=f"{self.name}_z0" if self.name else f"{self.__class__.__name__}_z0",
        )
        self.params["S"] = self.S
        self.params["z0"] = self.z0
        self.grads["S"] = None
        self.grads["z0"] = None

    def _initialize_z0_from_data(self, x):
        """Initialize z0 to the median of input data if not yet initialized."""
        if self._z0_initialized:
            return

        array = np.asarray(x, dtype=np.float64)
        if array.size > 0:
            median_val = float(np.median(array))
            self.z0.data = np.array([median_val], dtype=np.float64)
            self._z0_initialized = True
            if self.verbose:
                print(f"  {self.name}.z0 initialized to median={median_val:.6f}")

    def _as_tensor(self, x) -> Tensor:
        return x if isinstance(x, Tensor) else Tensor(x)

    @property
    def effective_S(self) -> Tensor:
        return Tensor.from_torch(_effective_scale(self.S._tensor, self.learnable_S).detach(), name=f"{self.name}_effective_S")

    def _theta(self, x) -> tuple[Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Initialize z0 from data on first forward pass if not user-provided
        self._initialize_z0_from_data(x)
        
        x = self._as_tensor(x)
        scale, theta, two_theta, active = _theta_components(x._tensor, self.S._tensor, self.z0._tensor, self.learnable_S)
        self.theta = Tensor.from_torch(theta.detach())
        return x, scale, theta, two_theta, active
