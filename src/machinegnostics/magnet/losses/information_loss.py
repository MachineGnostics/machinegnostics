"""Information loss for MAGNET."""

from __future__ import annotations

import torch

from ..core.tensor import Tensor
from ._gnostic_terms import _clip_gradient, compute_terms
from .base import Loss, prepare_tensors


class InformationLossFunction(torch.autograd.Function):
    """Custom autograd kernel for binary information from gnostic terms.

    Mathematical Notes:
        Forward computation
        -------------------
        1. Form ``θ = (y_pred - y_true) / S``.
        2. Compute ``hi = tanh(2θ)``.
        3. Convert to the binary proxy probability ``p = (1 - hi) / 2``.
        4. Return the mean binary entropy
           ``L = mean(-(p log p + (1 - p) log(1 - p)))``.

        Backward computation
        --------------------
        ``dH/dp = log((1 - p) / p)``, ``dp/dhi = -1/2``, and
        ``dhi/dθ = 2 fi²``. Therefore
        ``∂H/∂θ = -fi² log((1 - p) / p)`` and
        ``∂L/∂y_pred = -fi² log((1 - p) / p) / (N S)``.

        Numerical stability
        -------------------
        ``p`` is clamped to ``[EPS, 1 - EPS]`` before applying the logarithms.
        The custom backward returns zero once ``p`` or ``2θ`` hits a stability
        clamp, and clips the final prediction gradient to the project limit.
    """

    @staticmethod
    def forward(ctx, y_pred, y_true, scale):
        terms = compute_terms(y_pred - y_true, scale=scale)
        ctx.scale_value = float(terms["scale"].item())
        ctx.normalizer = max(y_pred.numel(), 1)
        ctx.save_for_backward(terms["fi"], terms["p_i"], terms["p_active"])
        return torch.mean(terms["information"])

    @staticmethod
    def backward(ctx, grad_output):
        fi, p_i, p_active = ctx.saved_tensors
        delta_theta = -fi.square() * (torch.log1p(-p_i) - torch.log(p_i)) * p_active
        grad_pred = grad_output * delta_theta / (ctx.normalizer * ctx.scale_value)
        return _clip_gradient(grad_pred), None, None


class InformationLoss(Loss):
    """Minimize binary information induced by the gnostic characteristics."""

    def __init__(self, S: float = 1.0, name: str | None = None, verbose: bool = False):
        super().__init__(name=name, verbose=verbose)
        self.S = S

    def forward(self, y_pred, y_true) -> Tensor:
        """Return the batch-mean information objective.

        Mathematical Notes:
            ``y_true`` provides the fixed loss center and ``S`` stays manual.
            The custom backward differentiates the stabilized binary entropy only
            with respect to ``y_pred``.
        """
        y_pred, y_true = prepare_tensors(y_pred, y_true)
        loss = InformationLossFunction.apply(y_pred._tensor, y_true._tensor, self.S)
        return Tensor.from_torch(loss)
