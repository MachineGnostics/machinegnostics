"""Infidelity loss for MAGNET."""

from __future__ import annotations

import torch

from ..core.tensor import Tensor
from ._gnostic_terms import _clip_gradient, compute_terms
from .base import Loss, prepare_tensors


class InfidelityLossFunction(torch.autograd.Function):
    """Custom autograd kernel for ``mean(cosh(2θ))``.

    Mathematical Notes:
        Forward computation
        -------------------
        The loss evaluates ``fj = cosh(2θ)`` with ``θ = (y_pred - y_true) / S``
        and returns ``L = mean(fj)``.

        Backward computation
        --------------------
        Because ``∂fj/∂θ = 2 fj hi``, the prediction gradient is
        ``∂L/∂y_pred = 2 fj hi / (N S)``.

        Numerical stability
        -------------------
        ``2θ`` is clipped before evaluating hyperbolic terms and ``fj`` is
        capped by ``MAX_MAGNITUDE`` inside ``compute_terms``. Saturated elements
        contribute zero gradient.
    """

    @staticmethod
    def forward(ctx, y_pred, y_true, scale):
        terms = compute_terms(y_pred - y_true, scale=scale)
        ctx.scale_value = float(terms["scale"].item())
        ctx.normalizer = max(y_pred.numel(), 1)
        ctx.save_for_backward(terms["fj"], terms["hi"], terms["fj_active"])
        return torch.mean(terms["fj"])

    @staticmethod
    def backward(ctx, grad_output):
        fj, hi, fj_active = ctx.saved_tensors
        delta_theta = 2.0 * fj * hi * fj_active
        grad_pred = grad_output * delta_theta / (ctx.normalizer * ctx.scale_value)
        return _clip_gradient(grad_pred), None, None


class InfidelityLoss(Loss):
    """Penalize inverse fidelity via ``mean(cosh(2θ))``."""

    def __init__(self, S: float = 1.0, name: str | None = None, verbose: bool = False):
        super().__init__(name=name, verbose=verbose)
        self.S = S

    def forward(self, y_pred, y_true) -> Tensor:
        """Return the batch-mean inverse-fidelity objective.

        Mathematical Notes:
            ``S`` and ``y_true`` are treated as fixed inputs. The custom
            backward therefore returns only the stable prediction gradient for
            ``L = mean(cosh(2(y_pred - y_true)/S))``.
        """
        y_pred, y_true = prepare_tensors(y_pred, y_true)
        loss = InfidelityLossFunction.apply(y_pred._tensor, y_true._tensor, self.S)
        return Tensor.from_torch(loss)
