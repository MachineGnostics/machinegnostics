"""Hj activation for MAGNET."""

from __future__ import annotations

import numpy as np
import torch

from ..core.tensor import Tensor
from ._centered import CenteredGnosticActivation, MAX_MAGNITUDE


def hj(x, S: float = 1.0, z0: float = 0.0) -> np.ndarray:
    """Return ``sinh(2θ)`` for array-like input."""
    array = np.asarray(x, dtype=np.float64)
    theta = (array - z0) / max(abs(float(S)), 1e-6)
    return np.clip(np.sinh(np.clip(2.0 * theta, -30.0, 30.0)), -MAX_MAGNITUDE, MAX_MAGNITUDE)


class Hj(CenteredGnosticActivation):
    """Complementary irrelevance activation based on ``sinh(2θ)``.

    ``Hj`` is the complementary characteristic to :class:`Hi`; their ratio
    follows ``hj / fj = hi`` when the values are inside the unclipped range.
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
        output = torch.clamp(torch.sinh(two_theta), min=-MAX_MAGNITUDE, max=MAX_MAGNITUDE)
        self.last_output = Tensor.from_torch(output)
        return Tensor.from_torch(output)
