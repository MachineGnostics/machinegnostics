"""Fj activation for MAGNET."""

from __future__ import annotations

import numpy as np
import torch

from ..core.tensor import Tensor
from ._centered import CenteredGnosticActivation, MAX_MAGNITUDE


def fj(x, S: float = 1.0, z0: float = 0.0) -> np.ndarray:
    """Return ``cosh(2θ)`` for array-like input."""
    array = np.asarray(x, dtype=np.float64)
    theta = (array - z0) / max(abs(float(S)), 1e-6)
    return np.clip(np.cosh(np.clip(2.0 * theta, -30.0, 30.0)), 1.0, MAX_MAGNITUDE)


class Fj(CenteredGnosticActivation):
    """Complementary fidelity activation based on ``cosh(2θ)``.

    ``Fj`` is the reciprocal characteristic of :class:`Fi`, i.e.
    ``fj = 1 / fi`` after numerical clamping.
    """

    def __init__(
        self,
        learnable_S: bool = True,
        learnable_z0: bool = True,
        initial_S: float = 1.0,
        initial_z0: float = 0.0,
        name: str | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            learnable_S=learnable_S,
            learnable_z0=learnable_z0,
            initial_S=initial_S,
            initial_z0=initial_z0,
            name=name,
            verbose=verbose,
        )

    def forward(self, x, training: bool = True) -> Tensor:
        x, _, two_theta = self._theta(x)
        output = torch.clamp(torch.cosh(two_theta), min=1.0, max=MAX_MAGNITUDE)
        self.last_output = Tensor.from_torch(output)
        return Tensor.from_torch(output)
