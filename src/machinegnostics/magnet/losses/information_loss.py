"""Information loss for MAGNET."""

from __future__ import annotations

import torch

from ..core.tensor import Tensor
from ._gnostic_terms import _clip_gradient, compute_gnostic_terms, resolve_gnostic_scale
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
        terms = compute_gnostic_terms(y_pred - y_true, scale=scale)
        ctx.scale_value = float(terms["scale"].item())
        ctx.normalizer = max(y_pred.numel(), 1)
        ctx.save_for_backward(terms["fi"], terms["p_i"], terms["p_active"])
        return torch.mean(terms["information"])

    @staticmethod
    def backward(ctx, grad_output):
        fi, p_i, p_active = ctx.saved_tensors
        delta_theta = -fi.square() * (torch.log1p(-p_i) - torch.log(p_i)) * p_active
        grad_pred = - grad_output * delta_theta / (ctx.normalizer * ctx.scale_value)
        return _clip_gradient(grad_pred), None, None


class InformationLoss(Loss):
    """Measure information content induced by gnostic concept fidelity.

    ``InformationLoss`` quantifies how much information remains in the
    prediction residuals after mapping them into MAGNET's gnostic terms. A
    common Shannon-like fidelity form is ``L = mean(fi log(fi))`` with
    ``fi = sech(2θ)``. The current implementation uses a stabilized binary
    entropy proxy derived from ``hi`` so the loss stays well-behaved during
    optimization while retaining an information-theoretic interpretation.

    Parameters
    ----------
    S : float or str, optional
        Scale parameter to normalize prediction errors before evaluating
        information content. Use ``"auto"`` to derive a local scale from the
        current batch via ``ScaleParam``.
    name : str or None, optional
        Optional display name for summaries and diagnostics.
    verbose : bool, optional
        If ``True``, enable verbose logging for the loss instance.

    Notes
    -----
    Gnostic Concept:
        Information-style objectives help distinguish between confident
        low-entropy predictions and ambiguous high-entropy predictions in the
        concept space learned by MAGNET.

    Use Cases:
        Use ``InformationLoss`` when you want a smoother information-theoretic
        signal than direct fidelity or inverse-fidelity penalties provide.

    Examples
    --------
    >>> import numpy as np
    >>> from machinegnostics.magnet import InformationLoss, Tensor
    >>> y_pred = Tensor(np.array([[0.1], [0.2]]), requires_grad=True)
    >>> y_true = Tensor(np.zeros((2, 1)))
    >>> np.isfinite(float(InformationLoss()(y_pred, y_true)))
    True
    """

    def __init__(self, S: float | str = "auto", name: str | None = None, verbose: bool = False):
        super().__init__(name=name, verbose=verbose)
        self.S = S
        self.S_local = S

    def forward(self, y_pred, y_true) -> Tensor:
        """Return the batch-mean information objective.

        Parameters
        ----------
        y_pred : array-like or Tensor
            Predicted values whose residual information should be measured.
        y_true : array-like or Tensor
            Reference targets defining the gnostic concept center.

        Returns
        -------
        Tensor
            Scalar information-style loss computed from stabilized gnostic
            probabilities.

        Notes
        -----
        ``y_true`` provides the fixed loss center and ``S`` may be numeric or
        ``"auto"``. In auto mode, a batch-local scale is derived before
        evaluating the stabilized information surrogate. The custom backward
        differentiates the stabilized information surrogate only with respect
        to ``y_pred``.
        """
        y_pred, y_true = prepare_tensors(y_pred, y_true)
        self.S_local = resolve_gnostic_scale(y_pred._tensor - y_true._tensor, self.S)
        loss = InformationLossFunction.apply(y_pred._tensor, y_true._tensor, self.S_local)
        return Tensor.from_torch(loss)
