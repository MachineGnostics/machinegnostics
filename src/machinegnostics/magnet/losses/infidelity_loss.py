"""Numerically stable infidelity loss for MAGNET."""

from __future__ import annotations

import torch

from ..core.tensor import Tensor
from ._gnostic_terms import EPS, compute_terms
from .base import Loss, prepare_tensors


class InfidelityLoss(Loss):
    """Minimize inverse fidelity via ``mean(1 / fi)``.

    ``fi`` is clamped to keep the loss finite and its upstream gradients are
    clipped to protect training from exploding updates.
    """

    def __init__(self, S: float = 1.0, name: str | None = None, verbose: bool = False):
        super().__init__(name=name, verbose=verbose)
        self.S = S

    def forward(self, y_pred, y_true) -> Tensor:
        y_pred, y_true = prepare_tensors(y_pred, y_true)
        terms = compute_terms(y_pred._tensor - y_true._tensor, scale=self.S)
        loss = torch.mean(1.0 / torch.clamp(terms["fi"], min=EPS))
        return Tensor.from_torch(loss)
