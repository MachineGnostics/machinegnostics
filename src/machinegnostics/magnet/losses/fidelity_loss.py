"""Fidelity loss for MAGNET."""

from __future__ import annotations

import torch

from ..core.tensor import Tensor
from ._gnostic_terms import _clip_gradient, compute_terms
from .base import Loss, prepare_tensors


class FidelityLossFunction(torch.autograd.Function):
    """Custom autograd kernel for ``mean(1 - fi)``.

    Mathematical Notes:
        Forward computation
        -------------------
        1. Form the residual-centered coordinate ``θ = (y_pred - y_true) / S``.
        2. Evaluate ``fi = sech(2θ)``.
        3. Return ``L = mean(1 - fi)``.

        Backward computation
        --------------------
        Since ``∂fi/∂θ = -2 fi hi``, the loss derivative is
        ``∂L/∂θ = 2 fi hi / N`` where ``N`` is the number of averaged elements.
        Because ``θ`` depends on predictions as ``θ = (y_pred - y_true) / S``,
        ``∂L/∂y_pred = 2 fi hi / (N S)``.

        Numerical stability
        -------------------
        The implementation clips ``2θ`` and clamps ``fi`` away from zero. Any
        saturated branch contributes zero gradient, and the final prediction
        gradient is clipped to the project-wide maximum magnitude.
    """

    @staticmethod
    def forward(ctx, y_pred, y_true, scale):
        terms = compute_terms(y_pred - y_true, scale=scale)
        ctx.scale_value = float(terms["scale"].item())
        ctx.normalizer = max(y_pred.numel(), 1)
        ctx.save_for_backward(terms["fi"], terms["hi"], terms["fi_active"])
        return torch.mean(terms["estimating_entropy"])

    @staticmethod
    def backward(ctx, grad_output):
        fi, hi, fi_active = ctx.saved_tensors
        delta_theta = 2.0 * fi * hi * fi_active
        grad_pred = grad_output * delta_theta / (ctx.normalizer * ctx.scale_value)
        return _clip_gradient(grad_pred), None, None


class FidelityLoss(Loss):
    """Minimize estimating entropy via ``mean(1 - fi)``.

    ``FidelityLoss`` reaches its optimum of ``0`` when predictions match the
    targets exactly and the residual-centered fidelity characteristic is 1.
    """

    def __init__(self, S: float = 1.0, name: str | None = None, verbose: bool = False):
        super().__init__(name=name, verbose=verbose)
        self.S = S

    def forward(self, y_pred, y_true) -> Tensor:
        """Return the batch-mean fidelity objective.

        Mathematical Notes:
            ``y_true`` supplies the fixed center ``z0`` for the loss and never
            receives gradients. ``S`` is a manual hyperparameter, so the custom
            backward only returns ``dL/dy_pred`` for
            ``L = mean(1 - sech(2(y_pred - y_true)/S))``.
        """
        y_pred, y_true = prepare_tensors(y_pred, y_true)
        loss = FidelityLossFunction.apply(y_pred._tensor, y_true._tensor, self.S)
        return Tensor.from_torch(loss)
