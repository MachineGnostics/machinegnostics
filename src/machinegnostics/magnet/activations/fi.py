"""Fi activation for MAGNET.

Fi models gnostic fidelity using the hyperbolic secant of the normalized
residual around a learnable center and scale.
"""

from __future__ import annotations

import numpy as np
import torch

from ..core.tensor import Tensor
from ._centered import CenteredGnosticActivation, EPS


def fi(x, S: float = 1.0, z0: float = 0.0) -> np.ndarray:
    """Return ``sech(2θ)`` for array-like input.

    ``θ = (x - z₀) / S`` and ``sech(2θ) = 1 / cosh(2θ)``.
    """
    array = np.asarray(x, dtype=np.float64)
    theta = (array - z0) / max(abs(float(S)), EPS)
    two_theta = np.clip(2.0 * theta, -30.0, 30.0)
    return np.clip(1.0 / np.cosh(two_theta), EPS, 1.0)


class Fi(CenteredGnosticActivation):
    """Fidelity activation based on ``sech(2θ)``.

    Mathematical Foundation
    -----------------------
    ``Fi = sech(2θ) = 2 / (e^(2θ) + e^(-2θ))``

    where ``θ = (x - z₀) / S``.

    The output stays in ``(0, 1]`` and peaks at ``θ = 0``.
    Together with :class:`Hi`, it satisfies the conservation identity
    ``fi² + hi² = 1``.
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
        output = torch.clamp(1.0 / torch.cosh(two_theta), min=EPS, max=1.0)
        self.last_output = Tensor.from_torch(output)
        return Tensor.from_torch(output)
