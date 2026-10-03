"""Fj activation for MAGNET.

This module contains the dedicated implementation of MAGNET's
complementary fidelity characteristic. ``Fj`` mirrors ``Fi`` while
expanding large deviations rather than suppressing them.
"""

from __future__ import annotations

import numpy as np
import torch

from ..core.tensor import Tensor
from ._centered import EPS, MAX_MAGNITUDE, CenteredGnosticActivation, _effective_scale, _scalar_parameter_gradient, _theta_components


def fj(x, S: float = 1.0, z0: float = 0.0) -> np.ndarray:
    """Evaluate the complementary fidelity characteristic ``cosh(2θ)``."""
    array = np.asarray(x, dtype=np.float64)
    theta = (array - z0) / max(abs(float(S)), EPS)
    two_theta = np.clip(2.0 * theta, -30.0, 30.0)
    return np.clip(np.cosh(two_theta), 1.0, MAX_MAGNITUDE)


class FjFunction(torch.autograd.Function):
    """Custom autograd kernel for ``fj = cosh(2θ)``.

    Mathematical Notes:
        Forward computation
        -------------------
        ``θ = (x - z0) / S`` and ``fj = cosh(2θ) = 1 / fi``.

        Backward computation
        --------------------
        With ``hi = tanh(2θ)``, ``∂fj/∂θ = 2 fj hi``.
        The centered-chain rule then gives
        ``dL/dx = δθ / S``,
        ``dL/dS_raw = -(1 - S/2) * Σ(δθ * θ)``, and
        ``dL/dz0 = -(1/S) * Σ(δθ)`` where ``δθ = grad_output * 2 fj hi``.

        Numerical stability
        -------------------
        ``2θ`` is clipped and ``fj`` is capped by ``MAX_MAGNITUDE``. The custom
        backward suppresses gradients once either saturation path is active.
    """

    @staticmethod
    def forward(ctx, x, scale_parameter, z0, learnable_S):
        scale, theta, two_theta, active = _theta_components(x, scale_parameter, z0, learnable_S)
        hi = torch.tanh(two_theta)
        fj_raw = torch.cosh(two_theta)
        output = torch.clamp(fj_raw, min=1.0, max=MAX_MAGNITUDE)
        fj_active = active * (fj_raw < MAX_MAGNITUDE).to(dtype=x.dtype)
        ctx.learnable_S = learnable_S
        ctx.save_for_backward(theta, scale, output, hi, fj_active, scale_parameter, z0)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        theta, scale, fj, hi, fj_active, scale_parameter, z0 = ctx.saved_tensors
        delta_theta = grad_output * (2.0 * fj * hi) * fj_active
        grad_x = delta_theta / scale
        grad_scale = None
        if ctx.needs_input_grad[1]:
            gate = 1.0 - scale / 2.0 if ctx.learnable_S else -1.0 / scale
            grad_scale = _scalar_parameter_gradient(-(gate * delta_theta * theta), scale_parameter)
        grad_z0 = None
        if ctx.needs_input_grad[2]:
            grad_z0 = _scalar_parameter_gradient(-(delta_theta / scale), z0)
        return grad_x, grad_scale, grad_z0, None


class Fj(CenteredGnosticActivation):
    """Learnable complementary fidelity activation ``cosh(2θ)``.

    ``Fj`` is the reciprocal counterpart to ``Fi`` after numerical
    clamping. It is useful when the model should amplify deviation from
    the learned center rather than reward alignment.
    """

    def __init__(
        self,
        learnable_S: bool = True,
        learnable_z0: bool = True,
        initial_S: float = 1.0,
        initial_z0: float = 0.0,
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

    def forward(self, x, training: bool = True) -> Tensor:
        """Transform inputs into complementary fidelity values.

        Mathematical Notes:
            ``Fj`` evaluates ``cosh(2θ)`` using the same centered geometry as
            ``Fi``. Its custom backward propagates ``∂fj/∂θ = 2 fj hi`` and uses
            the sigmoid-gated scale rule for learnable scales.
        """
        x = self._as_tensor(x)
        output = FjFunction.apply(x._tensor, self.S._tensor, self.z0._tensor, self.learnable_S)
        with torch.no_grad():
            scale = _effective_scale(self.S._tensor, self.learnable_S)
            theta = (x._tensor - self.z0._tensor) / scale
        self.theta = Tensor.from_torch(theta.detach())
        self.last_output = Tensor.from_torch(output)
        return Tensor.from_torch(output)
