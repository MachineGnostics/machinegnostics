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
        return torch.sum(terms["hj"] ** 2)

    @staticmethod
    def backward(ctx, grad_output):
        hj, fj, hj_active = ctx.saved_tensors
        delta_theta = 4.0 * hj * fj * hj_active
        grad_pred = grad_output * delta_theta / (ctx.normalizer * ctx.scale_value)
        return _clip_gradient(grad_pred), None, None


class ISSLoss(Loss):
    """Invariance Squared Sum (ISS) loss for MAGNET.

    ``ISSLoss`` is MAGNET's invariance-based objective that quantifies how
    much residual remains invariant under the gnostic transformation. The
    conceptual form emphasizes quantifying irrelevance through the surrogate
    ``mean(hj²)`` with ``hj = sinh(2θ)`` where ``θ = (y_pred - y_true) / S``
    is the centered residual. This numerically stable formulation preserves
    strong off-center penalization without introducing singularities.

    Parameters
    ----------
    S : float, optional
        Fixed residual scale used to normalize prediction errors before the
        inverse-style surrogate is evaluated.
    name : str or None, optional
        Optional display name for summaries and diagnostics.
    verbose : bool, optional
        If ``True``, enable verbose logging for loss debugging.

    Notes
    -----
    Gnostic Concept:
        ``ISSLoss`` emphasizes quantifying irrelevance, making it useful when
        large directional residuals should receive stronger gradients than a
        bounded fidelity objective would supply.

    Numerical Stability:
        The gnostic surrogate avoids the singularity that a literal
        ``1 / θ²`` objective would have at perfect alignment.

    Examples
    --------
    >>> import numpy as np
    >>> from machinegnostics.magnet import ISSLoss, Tensor
    >>> y_pred = Tensor(np.zeros((2, 1)), requires_grad=True)
    >>> y_true = Tensor(np.zeros((2, 1)))
    >>> float(ISSLoss()(y_pred, y_true))
    0.0
    """

    def __init__(self, S: float = 1.0, name: str | None = None, verbose: bool = False):
        super().__init__(name=name, verbose=verbose)
        self.S = S

    def forward(self, y_pred, y_true) -> Tensor:
        """Return the batch inverse-style residual surrogate.

        Parameters
        ----------
        y_pred : array-like or Tensor
            Predicted values to evaluate in centered residual coordinates.
        y_true : array-like or Tensor
            Reference targets defining the residual center.

        Returns
        -------
        Tensor
            Scalar loss equal to ``mean(sinh²(2(y_pred - y_true) / S))``.

        Notes
        -----
        ``y_true`` acts as the fixed center and ``S`` is a fixed scale
        hyperparameter. The custom backward returns only ``dL/dy_pred``.
        """
        y_pred, y_true = prepare_tensors(y_pred, y_true)
        loss = ISSLossFunction.apply(y_pred._tensor, y_true._tensor, self.S)
        return Tensor.from_torch(loss)
