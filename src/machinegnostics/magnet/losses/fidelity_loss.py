"""Fidelity loss for MAGNET."""

from __future__ import annotations

import torch

from ..core.tensor import Tensor
from ._gnostic_terms import compute_terms
from .base import Loss, prepare_tensors


class FidelityLoss(Loss):
    """Maximize estimating fidelity by minimizing ``-mean(fi)``.

    ``FidelityLoss`` is best when training should explicitly reward
    alignment with the learned gnostic center. At perfect alignment the
    residual is zero, ``fi`` is 1, and the loss reaches its optimum of
    ``-1``.
    """

    def __init__(self, S: float = 1.0, name: str | None = None, verbose: bool = False):
        super().__init__(name=name, verbose=verbose)
        self.S = S

    def forward(self, y_pred, y_true) -> Tensor:
        """Return the batch-mean fidelity objective.

        Parameters
        ----------
        y_pred : Tensor or array-like
            Predicted values.
        y_true : Tensor or array-like
            Target values with the same shape as ``y_pred``.

        Returns
        -------
        Tensor
            Scalar tensor in the interval ``[-1, 0]``.
        """
        y_pred, y_true = prepare_tensors(y_pred, y_true)
        terms = compute_terms(y_pred._tensor - y_true._tensor, scale=self.S)
        loss = -torch.mean(terms['fi'])
        return Tensor.from_torch(loss)
