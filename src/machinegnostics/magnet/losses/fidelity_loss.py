"""Fidelity loss for MAGNET."""

from __future__ import annotations

import torch

from ..core.tensor import Tensor
from ._gnostic_terms import _clip_gradient, compute_terms
from .base import Loss, prepare_tensors


class FidelityLossFunction(torch.autograd.Function):
    """Custom autograd kernel for ``-mean(fi)``.

    Mathematical Notes:
        Forward computation
        -------------------
        1. Form the residual-centered coordinate ``θ = (y_pred - y_true) / S``.
        2. Evaluate ``fi = sech(2θ)``.
        3. Return ``L = -mean(fi)``.

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
        return torch.mean(terms["fi"])

    @staticmethod
    def backward(ctx, grad_output):
        fi, hi, fi_active = ctx.saved_tensors
        delta_theta = 2.0 * fi * hi * fi_active
        grad_pred = grad_output * delta_theta / (ctx.normalizer * ctx.scale_value)
        return _clip_gradient(grad_pred), None, None


class FidelityLoss(Loss):
    """Maximize gnostic fidelity through the negative mean characteristic.

    ``FidelityLoss`` rewards predictions that align with the target-centered
    concept manifold. For each residual-centered coordinate
    ``θ = (y_pred - y_true) / S``, the loss evaluates the fidelity response
    ``fi = sech(2θ)`` and minimizes
    ``L = -mean(fi) = -mean(sech(2θ))``. The objective therefore lives in the
    interval ``[-1, 0)`` for finite residuals, reaches its optimum of ``-1``
    when every prediction lands exactly on the learned concept, and approaches
    ``0`` as fidelity collapses.

    Parameters
    ----------
    S : float, optional
        Fixed residual scale used to normalize prediction errors before
        measuring fidelity. Larger values widen the notion of concept
        agreement, while smaller values make the loss reward tight alignment.
    name : str or None, optional
        Optional display name for logging and summaries inside MAGNET training
        workflows.
    verbose : bool, optional
        If ``True``, enable verbose loss-level logging for debugging gnostic
        training behavior.

    Notes
    -----
    Gnostic Concept:
        High fidelity corresponds to strong concept concentration around the
        target manifold. Minimizing the negative mean fidelity directly rewards
        confident, centered predictions instead of penalizing them indirectly
        through ``1 - fi``.

    Examples
    --------
    >>> import numpy as np
    >>> from machinegnostics.magnet import FidelityLoss, Tensor
    >>> y_pred = Tensor(np.array([[0.0], [0.0]]), requires_grad=True)
    >>> y_true = Tensor(np.array([[0.0], [0.0]]))
    >>> float(FidelityLoss()(y_pred, y_true))
    -1.0
    """

    def __init__(self, S: float = 1.0, name: str | None = None, verbose: bool = False):
        super().__init__(name=name, verbose=verbose)
        self.S = S

    def forward(self, y_pred, y_true) -> Tensor:
        """Return the negative mean fidelity for a batch of predictions.

        Parameters
        ----------
        y_pred : array-like or Tensor
            Predicted values whose residuals will be evaluated against the
            target-centered gnostic fidelity characteristic.
        y_true : array-like or Tensor
            Reference targets that define the concept center for the loss.

        Returns
        -------
        Tensor
            Scalar MAGNET tensor equal to
            ``-mean(sech(2(y_pred - y_true) / S))``.

        Notes
        -----
        ``y_true`` supplies the fixed center ``z0`` for the loss and never
        receives gradients. ``S`` is a manual hyperparameter, so the custom
        backward only returns ``dL/dy_pred`` for the negative-fidelity form.
        """
        y_pred, y_true = prepare_tensors(y_pred, y_true)
        loss = FidelityLossFunction.apply(y_pred._tensor, y_true._tensor, self.S)
        return Tensor.from_torch(loss)
