"""RSS loss for MAGNET."""

from __future__ import annotations

import torch

from ..core.tensor import Tensor
from ._gnostic_terms import compute_terms
from .base import Loss, prepare_tensors


class RSSLoss(Loss):
    """Penalize the mean squared estimating irrelevance ``hi²``."""

    def __init__(self, S: float = 1.0, name: str | None = None, verbose: bool = False):
        super().__init__(name=name, verbose=verbose)
        self.S = S

    def forward(self, y_pred, y_true) -> Tensor:
        """Return the batch mean of the squared ``hi`` characteristic."""
        y_pred, y_true = prepare_tensors(y_pred, y_true)
        terms = compute_terms(y_pred._tensor - y_true._tensor, scale=self.S)
        loss = torch.mean(terms['hi'] ** 2)
        return Tensor.from_torch(loss)
