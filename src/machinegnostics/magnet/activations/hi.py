"""Hi activation for MAGNET.

This module contains the dedicated implementation of MAGNET's estimating
irrelevance characteristic. ``Hi`` shares the same learnable center and
scale structure as ``Fi`` but maps residual geometry into a signed,
bounded response.
"""

from __future__ import annotations

import numpy as np
import torch

from ..core.tensor import Tensor
from ._centered import CenteredGnosticActivation, EPS


def hi(x, S: float = 1.0, z0: float = 0.0) -> np.ndarray:
    """Evaluate the estimating irrelevance characteristic ``tanh(2θ)``."""
    array = np.asarray(x, dtype=np.float64)
    theta = (array - z0) / max(abs(float(S)), EPS)
    return np.tanh(np.clip(2.0 * theta, -30.0, 30.0))


class Hi(CenteredGnosticActivation):
    """Learnable estimating irrelevance activation ``tanh(2θ)``.

    ``Hi`` is the signed complement to ``Fi``. It measures directional
    deviation from the learned center while preserving a smooth,
    saturating response that remains bounded between -1 and 1.

    Attributes
    ----------
    S : Tensor
        Trainable or fixed positive scale parameter.
    z0 : Tensor
        Trainable or fixed center parameter.
    theta : Tensor
        Cached normalized deviation from the last forward pass.
    last_output : Tensor
        Cached activation output from the last forward pass.
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
        """Initialize the Hi activation layer."""
        super().__init__(
            learnable_S=learnable_S,
            learnable_z0=learnable_z0,
            initial_S=initial_S,
            initial_z0=initial_z0,
            name=name,
            verbose=verbose,
        )

    def forward(self, x, training: bool = True) -> Tensor:
        """Transform inputs into irrelevance values.

        Parameters
        ----------
        x : Tensor or array-like
            Input values or residuals to transform.
        training : bool, optional
            Compatibility flag for the MAGNET layer API.

        Returns
        -------
        Tensor
            Tensor with the same shape as ``x`` and values in ``[-1, 1]``.
        """
        x, _, two_theta = self._theta(x)
        output = torch.tanh(two_theta)
        self.last_output = Tensor.from_torch(output)
        return Tensor.from_torch(output)
