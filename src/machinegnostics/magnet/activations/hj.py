"""Hj activation for MAGNET.

This module contains the dedicated implementation of MAGNET's
quantifying irrelevance characteristic. ``Hj`` complements ``Hi`` by
producing an unbounded signed response that grows with deviation.
"""

from __future__ import annotations

import numpy as np
import torch

from ..core.tensor import Tensor
from ._centered import CenteredGnosticActivation, EPS, MAX_MAGNITUDE


def hj(x, S: float = 1.0, z0: float = 0.0) -> np.ndarray:
    """Evaluate the quantifying irrelevance characteristic ``sinh(2θ)``."""
    array = np.asarray(x, dtype=np.float64)
    theta = (array - z0) / max(abs(float(S)), EPS)
    two_theta = np.clip(2.0 * theta, -30.0, 30.0)
    return np.clip(np.sinh(two_theta), -MAX_MAGNITUDE, MAX_MAGNITUDE)


class Hj(CenteredGnosticActivation):
    """Learnable quantifying irrelevance activation ``sinh(2θ)``."""

    def __init__(
        self,
        learnable_S: bool = True,
        learnable_z0: bool = True,
        initial_S: float = 1.0,
        initial_z0: float = 0.0,
        name: str | None = None,
        verbose: bool = False,
    ):
        """Initialize the Hj activation layer."""
        super().__init__(
            learnable_S=learnable_S,
            learnable_z0=learnable_z0,
            initial_S=initial_S,
            initial_z0=initial_z0,
            name=name,
            verbose=verbose,
        )

    def forward(self, x, training: bool = True) -> Tensor:
        """Transform inputs into quantifying irrelevance values.

        Parameters
        ----------
        x : Tensor or array-like
            Input values or residuals to transform.
        training : bool, optional
            Compatibility flag for the MAGNET layer API.

        Returns
        -------
        Tensor
            Tensor with the same shape as ``x`` containing the clipped
            ``sinh(2θ)`` response.
        """
        x, _, two_theta = self._theta(x)
        output = torch.clamp(torch.sinh(two_theta), min=-MAX_MAGNITUDE, max=MAX_MAGNITUDE)
        self.last_output = Tensor.from_torch(output)
        return Tensor.from_torch(output)
