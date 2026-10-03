"""Ei activation for MAGNET.

This module contains the dedicated implementation of MAGNET's
entropy-like activation. ``Ei`` can operate in an estimating mode based
on ``Fi`` or a quantifying mode based on ``Fj``.
"""

from __future__ import annotations

import numpy as np
import torch

from ..core.tensor import Tensor
from ._centered import CenteredGnosticActivation, EPS, MAX_MAGNITUDE


def ei(x, S: float = 1.0, z0: float = 0.0, case: str = 'i') -> np.ndarray:
    """Evaluate the MAGNET entropy-like characteristic in NumPy."""
    array = np.asarray(x, dtype=np.float64)
    theta = (array - z0) / max(abs(float(S)), EPS)
    two_theta = np.clip(2.0 * theta, -30.0, 30.0)
    if case == 'i':
        fi_value = np.clip(1.0 / np.cosh(two_theta), EPS, 1.0)
        return 1.0 - fi_value
    if case == 'j':
        fj_value = np.clip(np.cosh(two_theta), 1.0, MAX_MAGNITUDE)
        return np.clip(fj_value - 1.0, 0.0, MAX_MAGNITUDE)
    raise ValueError("case must be 'i' or 'j'")


class Ei(CenteredGnosticActivation):
    """Learnable entropy-like activation derived from gnostic terms."""

    def __init__(
        self,
        learnable_S: bool = True,
        learnable_z0: bool = True,
        initial_S: float = 1.0,
        initial_z0: float = 0.0,
        case: str = 'i',
        name: str | None = None,
        verbose: bool = False,
    ):
        """Initialize the Ei activation layer."""
        super().__init__(
            learnable_S=learnable_S,
            learnable_z0=learnable_z0,
            initial_S=initial_S,
            initial_z0=initial_z0,
            name=name,
            verbose=verbose,
        )
        if case not in {'i', 'j'}:
            raise ValueError("case must be 'i' or 'j'")
        self.case = case

    def forward(self, x, training: bool = True) -> Tensor:
        """Transform inputs into the selected entropy-like response.

        Parameters
        ----------
        x : Tensor or array-like
            Input values or residuals to transform.
        training : bool, optional
            Compatibility flag for the MAGNET layer API.

        Returns
        -------
        Tensor
            Tensor with the same shape as ``x`` containing the selected
            non-negative entropy-like response.
        """
        x, _, two_theta = self._theta(x)
        if self.case == 'i':
            fi_value = torch.clamp(1.0 / torch.cosh(two_theta), min=EPS, max=1.0)
            output = 1.0 - fi_value
        else:
            fj_value = torch.clamp(torch.cosh(two_theta), min=1.0, max=MAX_MAGNITUDE)
            output = torch.clamp(fj_value - 1.0, min=0.0, max=MAX_MAGNITUDE)
        self.last_output = Tensor.from_torch(output)
        return Tensor.from_torch(output)
