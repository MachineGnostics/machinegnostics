"""Shared helpers for centered gnostic activations."""

from __future__ import annotations

import numpy as np
import torch

from ..core.tensor import Tensor
from .base import Activation

EPS = 1e-6
MAX_ABS_TWO_THETA = 30.0
MAX_MAGNITUDE = 1e6


class CenteredGnosticActivation(Activation):
    """Activation with learnable center ``z0`` and positive scale ``S``."""

    def __init__(
        self,
        *,
        learnable_S: bool = True,
        learnable_z0: bool = True,
        initial_S: float = 1.0,
        initial_z0: float = 0.0,
        name: str | None = None,
        verbose: bool = False,
    ):
        super().__init__(name=name, verbose=verbose)
        if initial_S <= 0:
            raise ValueError(f"initial_S must be positive, got {initial_S}")

        self.learnable_S = learnable_S
        self.learnable_z0 = learnable_z0
        self.S = Tensor(
            np.array([initial_S], dtype=np.float64),
            requires_grad=learnable_S,
            name=f"{self.name}_S" if self.name else f"{self.__class__.__name__}_S",
        )
        self.z0 = Tensor(
            np.array([initial_z0], dtype=np.float64),
            requires_grad=learnable_z0,
            name=f"{self.name}_z0" if self.name else f"{self.__class__.__name__}_z0",
        )
        self.params["S"] = self.S
        self.params["z0"] = self.z0
        self.grads["S"] = None
        self.grads["z0"] = None

    def _as_tensor(self, x) -> Tensor:
        return x if isinstance(x, Tensor) else Tensor(x)

    def _theta(self, x) -> tuple[Tensor, torch.Tensor, torch.Tensor]:
        x = self._as_tensor(x)
        x_tensor = x._tensor
        scale = torch.clamp(torch.abs(self.S._tensor), min=EPS)
        theta = (x_tensor - self.z0._tensor) / scale
        two_theta = torch.clamp(2.0 * theta, min=-MAX_ABS_TWO_THETA, max=MAX_ABS_TWO_THETA)
        self.theta = Tensor.from_torch(theta)
        return x, theta, two_theta


def _safe_scale_value(S: float) -> float:
    scale = abs(float(S))
    if scale < EPS:
        scale = EPS
    return scale
