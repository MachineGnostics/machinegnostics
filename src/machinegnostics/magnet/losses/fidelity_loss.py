"""Numerically stable fidelity loss for MAGNET."""

from __future__ import annotations

import torch

from ..core.tensor import Tensor
from ._gnostic_terms import EPS, compute_terms
from .base import Loss, prepare_tensors


class FidelityLoss(Loss):
    """Maximize fidelity by minimizing ``-mean(fi)``.

    With ``fi = sech(2θ)``, this loss is -1.0 at perfect alignment
    (where fi = 1) and approaches 0 as the residual moves away
    (where fi → 0). Minimizing this loss maximizes fidelity.

    Mathematical Foundation
    -----------------------
    L = -mean(fi)

    where fi = sech(2θ) = 2 / (e^(2θ) + e^(-2θ))

    Properties:
    - At perfect alignment: fi = 1.0, loss = -1.0 (best)
    - At poor alignment: fi → 0, loss → 0 (worst)
    - Loss range: [-1.0, 0]
    """

    def __init__(self, S: float = 1.0, name: str | None = None, verbose: bool = False):
        super().__init__(name=name, verbose=verbose)
        self.S = S

    def forward(self, y_pred, y_true) -> Tensor:
        """Compute fidelity loss.

        Parameters
        ----------
        y_pred : Tensor or array-like
            Predictions
        y_true : Tensor or array-like
            Ground truth

        Returns
        -------
        Tensor
            Scalar loss value in range [-1.0, 0]
        """
        y_pred, y_true = prepare_tensors(y_pred, y_true)
        terms = compute_terms(y_pred._tensor - y_true._tensor, scale=self.S)
        # Maximize fidelity: L = -mean(fi)
        loss = -torch.mean(terms["fi"])
        return Tensor.from_torch(loss)
