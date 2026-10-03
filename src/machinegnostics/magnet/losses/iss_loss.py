"""ISS loss for MAGNET."""

from __future__ import annotations

import torch

from ..core.tensor import Tensor
from ._gnostic_terms import _clip_gradient, compute_terms
from .base import Loss, prepare_tensors


class ISSLossFunction(torch.autograd.Function):
    """Custom autograd kernel for ``mean(hj²)``.

    Mathematical Notes:
        Forward computation
        -------------------
        ``hj = sinh(2θ)`` with ``θ = (y_pred - y_true) / S`` and
        ``L = mean(hj²)``.

        Backward computation
        --------------------
        Since ``∂hj/∂θ = 2 fj``, ``∂(hj²)/∂θ = 4 hj fj``.
        Therefore ``∂L/∂y_pred = 4 hj fj / (N S)``.

        Numerical stability
        -------------------
        The implementation uses the clipped-and-capped ``hj`` and ``fj`` terms
        and suppresses gradients through saturated elements.
    """

    @staticmethod
    def forward(ctx, y_pred, y_true, scale):
        terms = compute_terms(y_pred - y_true, scale=scale)
        ctx.scale_value = float(terms["scale"].item())
        ctx.normalizer = max(y_pred.numel(), 1)
        ctx.save_for_backward(terms["hj"], terms["fj"], terms["hj_active"])
        return torch.mean(terms["hj"] ** 2)

    @staticmethod
    def backward(ctx, grad_output):
        hj, fj, hj_active = ctx.saved_tensors
        delta_theta = 4.0 * hj * fj * hj_active
        grad_pred = grad_output * delta_theta / (ctx.normalizer * ctx.scale_value)
        return _clip_gradient(grad_pred), None, None


class ISSLoss(Loss):
    """Penalize the mean squared quantifying irrelevance ``hj²``."""

    def __init__(self, S: float = 1.0, name: str | None = None, verbose: bool = False):
        super().__init__(name=name, verbose=verbose)
        self.S = S

    def forward(self, y_pred, y_true) -> Tensor:
        """Return the batch mean of the squared ``hj`` characteristic.

        Mathematical Notes:
            ``y_true`` acts as the fixed center and ``S`` is a fixed scale
            hyperparameter. The custom backward returns only ``dL/dy_pred`` for
            ``L = mean(sinh²(2(y_pred - y_true)/S))``.
        """
        y_pred, y_true = prepare_tensors(y_pred, y_true)
        loss = ISSLossFunction.apply(y_pred._tensor, y_true._tensor, self.S)
        return Tensor.from_torch(loss)
