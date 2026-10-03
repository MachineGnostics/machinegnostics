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
        return torch.sum(terms["hi"] ** 2)

    @staticmethod
    def backward(ctx, grad_output):
        hi, fi, clip_mask = ctx.saved_tensors
        delta_theta = 4.0 * hi * fi.square() * clip_mask
        grad_pred = grad_output * delta_theta / (ctx.normalizer * ctx.scale_value)
        return _clip_gradient(grad_pred), None, None


class RSSLoss(Loss):
    """Relevance Squared Sum (RSS) loss for MAGNET.

    ``RSSLoss`` is MAGNET's relevance-based objective that measures how much
    of the residual is relevant within the gnostic concept space. The
    conceptual form is ``L = mean(θ²)`` with ``θ = (y_pred - y_true) / S``.
    This implementation uses the bounded surrogate ``mean(hi²)`` where
    ``hi = tanh(2θ)``, preserving the same zero-loss optimum while improving
    numerical stability for large residuals.

    Parameters
    ----------
    S : float, optional
        Scale Parameter to normalize prediction errors before evaluating the squared surrogate.
    name : str or None, optional
        Optional display name for summaries and debugging output.
    verbose : bool, optional
        If ``True``, enable verbose loss-level diagnostics.

    Notes
    -----
    Gnostic Concept:
        ``RSSLoss`` provides a squared residual penalty that remains aligned
        with MAGNET's centered-coordinate view of concept learning.

    Comparison to Gnostic Losses:
        Compared with fidelity or inverse-fidelity losses, RSS behaves more
        like a conventional regression objective while still using gnostic
        coordinates internally.

    Examples
    --------
    >>> import numpy as np
    >>> from machinegnostics.magnet import RSSLoss, Tensor
    >>> y_pred = Tensor(np.zeros((2, 1)), requires_grad=True)
    >>> y_true = Tensor(np.zeros((2, 1)))
    >>> float(RSSLoss()(y_pred, y_true))
    0.0
    """

    def __init__(self, S: float = 1.0, name: str | None = None, verbose: bool = False):
        super().__init__(name=name, verbose=verbose)
        self.S = S

    def forward(self, y_pred, y_true) -> Tensor:
        """Return the batch residual-squared surrogate.

        Parameters
        ----------
        y_pred : array-like or Tensor
            Predicted values whose centered residuals should be penalized.
        y_true : array-like or Tensor
            Reference targets defining zero residual.

        Returns
        -------
        Tensor
            Scalar loss equal to ``mean(tanh²(2(y_pred - y_true) / S))``.

        Notes
        -----
        The implementation is a bounded surrogate for the classical
        ``mean(θ²)`` form and only backpropagates through ``y_pred``.
        """
        y_pred, y_true = prepare_tensors(y_pred, y_true)
        loss = RSSLossFunction.apply(y_pred._tensor, y_true._tensor, self.S)
        return Tensor.from_torch(loss)
