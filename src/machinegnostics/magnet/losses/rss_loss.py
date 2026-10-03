"""RSS loss for MAGNET."""

from __future__ import annotations

import torch

from ..core.tensor import Tensor
from ._gnostic_terms import _clip_gradient, compute_terms
from .base import Loss, prepare_tensors


class RSSLossFunction(torch.autograd.Function):
    """Custom autograd kernel for ``mean(hi²)``.

    Mathematical Notes:
        Forward computation
        -------------------
        ``hi = tanh(2θ)`` with ``θ = (y_pred - y_true) / S`` and
        ``L = mean(hi²)``.

        Backward computation
        --------------------
        ``∂hi/∂θ = 2 fi²`` implies ``∂(hi²)/∂θ = 4 hi fi²``.
        Hence ``∂L/∂y_pred = 4 hi fi² / (N S)``.

        Numerical stability
        -------------------
        Once the shared ``2θ`` clip saturates, the stabilized forward path is
        flat and the custom backward returns zero.
    """

    @staticmethod
    def forward(ctx, y_pred, y_true, scale):
        terms = compute_terms(y_pred - y_true, scale=scale)
        ctx.scale_value = float(terms["scale"].item())
        ctx.normalizer = max(y_pred.numel(), 1)
        ctx.save_for_backward(terms["hi"], terms["fi"], terms["clip_mask"])
        return torch.mean(terms["hi"] ** 2)

    @staticmethod
    def backward(ctx, grad_output):
        hi, fi, clip_mask = ctx.saved_tensors
        delta_theta = 4.0 * hi * fi.square() * clip_mask
        grad_pred = grad_output * delta_theta / (ctx.normalizer * ctx.scale_value)
        return _clip_gradient(grad_pred), None, None


class RSSLoss(Loss):
    """Penalize the mean squared estimating irrelevance ``hi²``."""

    def __init__(self, S: float = 1.0, name: str | None = None, verbose: bool = False):
        super().__init__(name=name, verbose=verbose)
        self.S = S

    def forward(self, y_pred, y_true) -> Tensor:
        """Return the batch mean of the squared ``hi`` characteristic.

        Mathematical Notes:
            The forward pass computes ``mean(tanh²(2(y_pred - y_true)/S))``.
            ``S`` and ``y_true`` remain fixed, so only ``y_pred`` receives the
            custom backward gradient.
        """
        y_pred, y_true = prepare_tensors(y_pred, y_true)
        loss = RSSLossFunction.apply(y_pred._tensor, y_true._tensor, self.S)
        return Tensor.from_torch(loss)
