"""Residual-entropy loss for MAGNET."""

from __future__ import annotations

import torch

from ..core.tensor import Tensor
from ._gnostic_terms import _clip_gradient, compute_terms
from .base import Loss, prepare_tensors


class ResidualEntropyLossFunction(torch.autograd.Function):
    """Custom autograd kernel for ``mean(fj - fi)``.

    Mathematical Notes:
        Forward computation
        -------------------
        The loss evaluates the residual entropy surrogate ``L = mean(fj - fi)``
        with ``fj = cosh(2θ)`` and ``fi = sech(2θ)``.

        Backward computation
        --------------------
        ``∂fj/∂θ = 2 fj hi`` and ``∂fi/∂θ = -2 fi hi``. Combining them gives
        ``∂(fj - fi)/∂θ = 2 hi (fj + fi)``, so
        ``∂L/∂y_pred = 2 hi (fj + fi) / (N S)``.

        Numerical stability
        -------------------
        The forward path uses the shared clipped hyperbolic terms, and the
        backward path zeroes saturated branches before clipping the final
        prediction gradient.
    """

    @staticmethod
    def forward(ctx, y_pred, y_true, scale):
        terms = compute_terms(y_pred - y_true, scale=scale)
        ctx.scale_value = float(terms["scale"].item())
        ctx.normalizer = max(y_pred.numel(), 1)
        active = terms["fi_active"] * terms["fj_active"]
        ctx.save_for_backward(terms["fi"], terms["fj"], terms["hi"], active)
        return torch.mean(terms["fj"] - terms["fi"])

    @staticmethod
    def backward(ctx, grad_output):
        fi, fj, hi, active = ctx.saved_tensors
        delta_theta = 2.0 * hi * (fj + fi) * active
        grad_pred = grad_output * delta_theta / (ctx.normalizer * ctx.scale_value)
        return _clip_gradient(grad_pred), None, None


class ResidualEntropyLoss(Loss):
    """Minimize residual entropy defined as ``mean(fj - fi)``."""

    def __init__(self, S: float = 1.0, name: str | None = None, verbose: bool = False):
        super().__init__(name=name, verbose=verbose)
        self.S = S

    def forward(self, y_pred, y_true) -> Tensor:
        """Return the batch-mean residual-entropy objective.

        Mathematical Notes:
            The forward pass computes ``mean(cosh(2θ) - sech(2θ))`` with
            ``θ = (y_pred - y_true) / S``. ``S`` and ``y_true`` are treated as
            fixed inputs, so the custom backward returns only ``dL/dy_pred``.
        """
        y_pred, y_true = prepare_tensors(y_pred, y_true)
        loss = ResidualEntropyLossFunction.apply(y_pred._tensor, y_true._tensor, self.S)
        return Tensor.from_torch(loss)
