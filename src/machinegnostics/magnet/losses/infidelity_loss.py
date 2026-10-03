"""Infidelity loss for MAGNET."""

from __future__ import annotations

import torch

from ..core.tensor import Tensor
from ._gnostic_terms import EPS, compute_terms
from .base import Loss, prepare_tensors


class InfidelityLoss(Loss):
    """Penalize inverse fidelity via ``mean(1 / fi)``."""

    def __init__(self, S: float = 1.0, name: str | None = None, verbose: bool = False):
        super().__init__(name=name, verbose=verbose)
        self.S = S

    def forward(self, y_pred, y_true) -> Tensor:
        """Return the batch-mean inverse-fidelity objective.

        Parameters
        ----------
        y_pred : Tensor or array-like
            Predicted values.
        y_true : Tensor or array-like
            Target values with the same shape as ``y_pred``.

        Returns
        -------
        Tensor
            Scalar tensor greater than or equal to 1.
        """
        y_pred, y_true = prepare_tensors(y_pred, y_true)
        terms = compute_terms(y_pred._tensor - y_true._tensor, scale=self.S)
        loss = torch.mean(1.0 / torch.clamp(terms['fi'], min=EPS))
        return Tensor.from_torch(loss)
