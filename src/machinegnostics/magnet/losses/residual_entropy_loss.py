"""Residual-entropy loss for MAGNET.

This module implements the residual-entropy loss used in MAGNET, which measures how much information is lost when predictions deviate from the target-centered concept manifold. The loss provides a custom autograd function for efficient gradient computation.

Author: Nirmal Parmar
MAGNET (Machine Gnostics Neural Networks) framework.
"""

from __future__ import annotations

import torch

from ..core.tensor import Tensor
from ._gnostic_terms import _clip_gradient, compute_gnostic_terms, resolve_gnostic_scale
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
        terms = compute_gnostic_terms(y_pred - y_true, scale=scale)
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
    """Measure residual entropy from the loss of gnostic fidelity.

    ``ResidualEntropyLoss`` tracks how much information is lost when predictions
    move away from the concept manifold. MAGNET implements
    the residual entropy by ``mean(fj - fi)``, combining fidelity loss and
    inverse-fidelity growth so residual uncertainty grows more sharply as
    predictions leave the target neighborhood.

    Parameters
    ----------
    S : float or str, optional
        Scale parameter to normalize prediction errors before evaluating the
        residual entropy. Use ``"auto"`` to derive a local scale from the
        current batch via ``ScaleParam``.
    name : str or None, optional
        Optional display name for MAGNET summaries and debugging.
    verbose : bool, optional
        If ``True``, enable verbose logging for this loss instance.

    Notes
    -----
    Gnostic Concept:
        Residual entropy complements fidelity. High entropy means the model is
        uncertain or off-concept, while low entropy means predictions remain
        concentrated near the target manifold.

    Relationship to FidelityLoss:
        ``FidelityLoss`` rewards high ``fi`` directly; ``ResidualEntropyLoss``
        instead penalizes the entropy-style complement of fidelity.

    Examples
    --------
    >>> import numpy as np
    >>> from machinegnostics.magnet import ResidualEntropyLoss, Tensor
    >>> y_pred = Tensor(np.zeros((2, 1)), requires_grad=True)
    >>> y_true = Tensor(np.zeros((2, 1)))
    >>> float(ResidualEntropyLoss()(y_pred, y_true))
    0.0
    """

    def __init__(self, S: float | str = 1, name: str | None = None, verbose: bool = False):
        super().__init__(name=name, verbose=verbose)
        self.S = S
        self.S_local = S

    def forward(self, y_pred, y_true) -> Tensor:
        """Return the batch residual-entropy objective.

        Parameters
        ----------
        y_pred : array-like or Tensor
            Predicted values whose residual entropy should be measured.
        y_true : array-like or Tensor
            Reference targets defining the concept center.

        Returns
        -------
        Tensor
            Scalar loss equal to ``mean(cosh(2θ) - sech(2θ))`` with
            ``θ = (y_pred - y_true) / S``.

        Notes
        -----
        ``S`` may be numeric or ``"auto"``. In auto mode, a batch-local scale
        is derived before evaluating the entropy surrogate. The custom
        backward returns only ``dL/dy_pred``.
        """
        y_pred, y_true = prepare_tensors(y_pred, y_true)
        self.S_local = resolve_gnostic_scale(y_pred._tensor - y_true._tensor, self.S)
        loss = ResidualEntropyLossFunction.apply(y_pred._tensor, y_true._tensor, self.S_local)
        return Tensor.from_torch(loss)
