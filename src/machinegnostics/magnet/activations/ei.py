"""Ei activation for MAGNET.

This module contains the dedicated implementation of MAGNET's
entropy-like activation. ``Ei`` can operate in an estimating mode based
on ``Fi`` or a quantifying mode based on ``Fj``.
"""

from __future__ import annotations

import numpy as np
import torch

from ..core.tensor import Tensor
from ._centered import EPS, MAX_MAGNITUDE, CenteredGnosticActivation, _effective_scale, _scalar_parameter_gradient, _theta_components


def ei(x, S: float = 1.0, z0: float = 0.0, case: str = 'i') -> np.ndarray:
    """Evaluate the MAGNET entropy-like characteristic in NumPy."""
    array = np.asarray(x, dtype=np.float64)
    theta = (array - z0) / max(abs(float(S)), EPS)
    two_theta = np.clip(2.0 * theta, -30.0, 30.0)
    if case == 'i':
        fi_value = np.clip(1.0 / np.cosh(two_theta), EPS, 1.0)
        return 1.0 - fi_value
    if case == 'j':
        fj_value = np.clip(np.cosh(two_theta), 1.0, MAX_MAGNITUDE)
        return np.clip(fj_value - 1.0, 0.0, MAX_MAGNITUDE)
    raise ValueError("case must be 'i' or 'j'")


class EiFunction(torch.autograd.Function):
    """Custom autograd kernel for MAGNET's entropy-like activation.

    Mathematical Notes:
        Forward computation
        -------------------
        For case ``'i'``, ``ei = 1 - fi`` with ``fi = sech(2θ)``.
        For case ``'j'``, ``ei = fj - 1`` with ``fj = cosh(2θ)``.

        Backward computation
        --------------------
        Estimating mode uses ``∂ei/∂θ = 2 fi hi``.
        Quantifying mode uses ``∂ei/∂θ = 2 fj hi``.
        The centered-coordinate gradients for ``x``, ``S_raw``, and ``z0`` then
        follow exactly as in the other gnostic activations.

        Numerical stability
        -------------------
        The same clip and clamp safeguards used by Fi and Fj are applied here,
        and saturated branches return zero gradient.
    """

    @staticmethod
    def forward(ctx, x, scale_parameter, z0, learnable_S, case):
        scale, theta, two_theta, active = _theta_components(x, scale_parameter, z0, learnable_S)
        hi = torch.tanh(two_theta)
        fi_raw = 1.0 / torch.cosh(two_theta)
        fj_raw = torch.cosh(two_theta)
        ctx.learnable_S = learnable_S
        ctx.case = case
        if case == 'i':
            fi_value = torch.clamp(fi_raw, min=EPS, max=1.0)
            output = 1.0 - fi_value
            active_mask = active * (fi_raw > EPS).to(dtype=x.dtype)
            ctx.save_for_backward(theta, scale, fi_value, hi, active_mask, scale_parameter, z0)
            return output
        if case == 'j':
            fj_value = torch.clamp(fj_raw, min=1.0, max=MAX_MAGNITUDE)
            output = torch.clamp(fj_value - 1.0, min=0.0, max=MAX_MAGNITUDE)
            active_mask = active * (fj_raw < MAX_MAGNITUDE).to(dtype=x.dtype)
            ctx.save_for_backward(theta, scale, fj_value, hi, active_mask, scale_parameter, z0)
            return output
        raise ValueError("case must be 'i' or 'j'")

    @staticmethod
    def backward(ctx, grad_output):
        theta, scale, value, hi, active_mask, scale_parameter, z0 = ctx.saved_tensors
        delta_theta = grad_output * (2.0 * value * hi) * active_mask
        grad_x = delta_theta / scale
        grad_scale = None
        if ctx.needs_input_grad[1]:
            gate = 1.0 - scale / 2.0 if ctx.learnable_S else -1.0 / scale
            grad_scale = _scalar_parameter_gradient(-(gate * delta_theta * theta), scale_parameter)
        grad_z0 = None
        if ctx.needs_input_grad[2]:
            grad_z0 = _scalar_parameter_gradient(-(delta_theta / scale), z0)
        return grad_x, grad_scale, grad_z0, None, None


class Ei(CenteredGnosticActivation):
    """Learnable entropy-like activation derived from gnostic terms."""

    def __init__(
        self,
        learnable_S: bool = True,
        learnable_z0: bool = True,
        initial_S: float = 1.0,
        initial_z0: float = 0.0,
        case: str = 'i',
        name: str | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            learnable_S=learnable_S,
            learnable_z0=learnable_z0,
            initial_S=initial_S,
            initial_z0=initial_z0,
            name=name,
            verbose=verbose,
        )
        if case not in {'i', 'j'}:
            raise ValueError("case must be 'i' or 'j'")
        self.case = case

    def forward(self, x, training: bool = True) -> Tensor:
        """Transform inputs into the selected entropy-like response.

        Mathematical Notes:
            Case ``'i'`` implements ``ei = 1 - fi`` with derivative
            ``∂ei/∂θ = 2 fi hi``. Case ``'j'`` preserves the existing
            quantifying form ``ei = fj - 1`` with derivative
            ``∂ei/∂θ = 2 fj hi``. Both use the same sigmoid-gated scale and exact
            centered-coordinate backward rules.
        """
        x = self._as_tensor(x)
        output = EiFunction.apply(x._tensor, self.S._tensor, self.z0._tensor, self.learnable_S, self.case)
        with torch.no_grad():
            scale = _effective_scale(self.S._tensor, self.learnable_S)
            theta = (x._tensor - self.z0._tensor) / scale
        self.theta = Tensor.from_torch(theta.detach())
        self.last_output = Tensor.from_torch(output)
        return Tensor.from_torch(output)
