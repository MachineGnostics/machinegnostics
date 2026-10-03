"""Fj activation for MAGNET.

This module contains the dedicated implementation of MAGNET's
complementary fidelity characteristic. ``Fj`` mirrors ``Fi`` while
expanding large deviations rather than suppressing them.
"""

from __future__ import annotations

import numpy as np
import torch

from ..core.tensor import Tensor
from ._centered import CenteredGnosticActivation, EPS, MAX_MAGNITUDE


def fj(x, S: float = 1.0, z0: float = 0.0) -> np.ndarray:
    """Evaluate the complementary fidelity characteristic ``cosh(2θ)``."""
    array = np.asarray(x, dtype=np.float64)
    theta = (array - z0) / max(abs(float(S)), EPS)
    two_theta = np.clip(2.0 * theta, -30.0, 30.0)
    return np.clip(np.cosh(two_theta), 1.0, MAX_MAGNITUDE)


class Fj(CenteredGnosticActivation):
    """Learnable complementary fidelity activation ``cosh(2θ)``.

    ``Fj`` is the reciprocal counterpart to ``Fi`` after numerical
    clamping. It is useful when the model should amplify deviation from
    the learned center rather than reward alignment.
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
        """Initialize the Fj activation layer."""
        super().__init__(
            learnable_S=learnable_S,
            learnable_z0=learnable_z0,
            initial_S=initial_S,
            initial_z0=initial_z0,
            name=name,
            verbose=verbose,
        )

    def forward(self, x, training: bool = True) -> Tensor:
        """Transform inputs into complementary fidelity values.

        Parameters
        ----------
        x : Tensor or array-like
            Input values or residuals to transform.
        training : bool, optional
            Compatibility flag for the MAGNET layer API.

        Returns
        -------
        Tensor
            Tensor with the same shape as ``x`` and values in
            ``[1, MAX_MAGNITUDE]``.
        """
        x, _, two_theta = self._theta(x)
        output = torch.clamp(torch.cosh(two_theta), min=1.0, max=MAX_MAGNITUDE)
        self.last_output = Tensor.from_torch(output)
        return Tensor.from_torch(output)
