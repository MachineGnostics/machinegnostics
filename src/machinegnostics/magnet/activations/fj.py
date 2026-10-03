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
    """Amplify inverse fidelity relative to a learned concept center.

    ``Fj`` is MAGNET's inverse-fidelity activation. It evaluates
    ``fj = cosh(2θ)`` for ``θ = (x - z0) / S`` and therefore behaves as the
    reciprocal companion to :class:`Fi` through the identity ``fj = 1 / fi``
    before numerical clamping. Unlike ``Fi``, which concentrates features
    around the concept center, ``Fj`` grows as deviation increases and is
    useful when a model should magnify off-concept behavior.

    Parameters
    ----------
    learnable_S : bool, optional
        If ``True``, optimize the raw scale parameter while enforcing the
        effective positive scale ``S = 2σ(S_raw)`` during the forward pass.
    learnable_z0 : bool, optional
        If ``True``, learn the concept center from which inverse fidelity is
        measured.
    initial_S : float, optional
        Initial positive scale governing how quickly inverse fidelity grows.
    initial_z0 : float, optional
        Initial center used to define zero deviation.
    name : str or None, optional
        Optional display name for logging and parameter-history plots.
    verbose : bool, optional
        If ``True``, enable verbose activation-level diagnostics.

    Attributes
    ----------
    S : Tensor
        Stored raw scale parameter used to derive the effective positive scale.
    z0 : Tensor
        Learnable concept center for the inverse-fidelity geometry.
    theta : Tensor
        Most recently computed centered coordinate.
    last_output : Tensor
        Most recent inverse-fidelity response.

    Notes
    -----
    Gnostic Concept:
        ``Fj`` quantifies how strongly an input departs from the target
        concept. Large values indicate strong off-concept evidence and can be
        used in loss functions that penalize poor concept alignment.

    Derivative Relationship:
        ``Fj`` shares the same directional term ``hi`` used by
        :class:`Hi`, giving ``∂fj/∂θ = 2 fj hi``.

    Use Cases:
        Use ``Fj`` when training or analysis should emphasize residual growth,
        anomaly magnitude, or inverse confidence.

    Examples
    --------
    >>> import numpy as np
    >>> from machinegnostics.magnet import Fj
    >>> Fj(initial_S=1.0, initial_z0=0.0)(np.array([[0.0, 1.0]])).data.shape
    (1, 2)
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
        """Transform inputs into inverse-fidelity values.

        Parameters
        ----------
        x : array-like or Tensor
            Input features whose deviation from ``z0`` should be amplified.
        training : bool, optional
            Present for API compatibility. ``Fj`` uses the same deterministic
            response in training and inference.

        Returns
        -------
        Tensor
            Inverse-fidelity response in ``[1, MAX_MAGNITUDE]`` after
            stabilization.

        Notes
        -----
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
