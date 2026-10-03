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
    """Penalize inverse fidelity so off-concept predictions become expensive.

    ``InfidelityLoss`` minimizes the batch mean of the infidelity
    characteristic ``fj = cosh(2θ) = 1 / fi`` with
    ``θ = (y_pred - y_true) / S``. Because ``fj`` is bounded in [1, ∞), the loss emphasizes poor concept alignment much more aggressively than fidelity-only objectives.

    Parameters
    ----------
    S : float, optional
        Scale Parameter to normalize prediction errors before evaluating infidelity.
    name : str or None, optional
        Optional display name used by MAGNET summaries and logging.
    verbose : bool, optional
        If ``True``, enable verbose loss-level diagnostics.

    Notes
    -----
    Gnostic Concept:
        This loss penalizes concept mismatch through ``L = mean(1 / fi)``. Its
        minimum is ``1`` at perfect alignment and it increases without bound as
        fidelity collapses.

    Use Cases:
        Use ``InfidelityLoss`` when training should strongly discourage
        low-fidelity predictions or when you want a sharper penalty than
        negative fidelity provides.

    Examples
    --------
    >>> import numpy as np
    >>> from machinegnostics.magnet import InfidelityLoss, Tensor
    >>> y_pred = Tensor(np.zeros((2, 1)), requires_grad=True)
    >>> y_true = Tensor(np.zeros((2, 1)))
    >>> float(InfidelityLoss()(y_pred, y_true))
    1.0
    """

    def __init__(self, S: float = 1.0, name: str | None = None, verbose: bool = False):
        super().__init__(name=name, verbose=verbose)
        self.S = S

    def forward(self, y_pred, y_true) -> Tensor:
        """Return the batch-mean inverse-fidelity objective.

        Parameters
        ----------
        y_pred : array-like or Tensor
            Predicted values to compare against the target concept manifold.
        y_true : array-like or Tensor
            Reference targets defining the concept center for the loss.

        Returns
        -------
        Tensor
            Scalar loss equal to ``mean(cosh(2(y_pred - y_true) / S))``.

        Notes
        -----
        ``S`` and ``y_true`` are treated as fixed inputs. The custom backward
        therefore returns only the stable prediction gradient for the
        inverse-fidelity objective.
        """
        y_pred, y_true = prepare_tensors(y_pred, y_true)
        loss = InfidelityLossFunction.apply(y_pred._tensor, y_true._tensor, self.S)
        return Tensor.from_torch(loss)
