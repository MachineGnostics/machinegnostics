"""Fi activation for MAGNET.

This module contains the dedicated implementation of the estimating
fidelity characteristic used by MAGNET's gnostic activation family.
``Fi`` learns a center ``z0`` and, optionally, a scale ``S`` so the
activation can adapt to the residual geometry seen during training.

Author: Nirmal Parmar
Machine Gnostics (MAGNET) Implementation
"""

from __future__ import annotations

import numpy as np
import torch

from ..core.tensor import Tensor
from ._centered import EPS, CenteredGnosticActivation, _effective_scale, _scalar_parameter_gradient, _theta_components


def fi(x, S: float = 1.0, z0: float = 0.0) -> np.ndarray:
    """Evaluate the estimating fidelity characteristic ``sech(2θ)``.

    The helper mirrors the layer mathematics in plain NumPy form, which
    is useful for analysis, tests, and quick experimentation outside a
    full model graph.
    """
    array = np.asarray(x, dtype=np.float64)
    theta = (array - z0) / max(abs(float(S)), EPS)
    two_theta = np.clip(2.0 * theta, -30.0, 30.0)
    return np.clip(1.0 / np.cosh(two_theta), EPS, 1.0)


class FiFunction(torch.autograd.Function):
    """Custom autograd kernel for ``fi = sech(2θ)``.

    Mathematical Notes:
        Forward computation
        -------------------
        1. Map the stored scale parameter to the effective positive scale with
           ``S = 2σ(S_raw)`` when the scale is learnable.
        2. Form the centered normalized coordinate ``θ = (x - z0) / S``.
        3. Evaluate ``fi = sech(2θ) = 1 / cosh(2θ)``.

        Backward computation
        --------------------
        The characteristic derivative is ``∂fi/∂θ = -2 fi hi`` with
        ``hi = tanh(2θ)``. Therefore ``δθ = grad_output * (-2 fi hi)``.
        Using ``θ = (x - z0) / S`` gives ``∂θ/∂x = 1/S`` and
        ``∂θ/∂z0 = -1/S``. For the sigmoid-gated raw scale,
        ``∂θ/∂S_raw = -θ (1 - S/2)`` because ``dS/dS_raw = S(1 - S/2)``.
        The custom backward therefore returns
        ``dL/dS_raw = -(1 - S/2) * Σ(δθ * θ)`` and
        ``dL/dz0 = -(1/S) * Σ(δθ)``.

        Numerical stability
        -------------------
        ``2θ`` is clipped before hyperbolic evaluation and the output is clamped
        below by ``EPS``. When clipping or clamping becomes active, the custom
        backward returns zero through those saturated paths.
    """

    @staticmethod
    def forward(ctx, x, scale_parameter, z0, learnable_S):
        scale, theta, two_theta, active = _theta_components(x, scale_parameter, z0, learnable_S)
        hi = torch.tanh(two_theta)
        fi_raw = 1.0 / torch.cosh(two_theta)
        output = torch.clamp(fi_raw, min=EPS, max=1.0)
        fi_active = active * (fi_raw > EPS).to(dtype=x.dtype)
        ctx.learnable_S = learnable_S
        ctx.save_for_backward(theta, scale, output, hi, fi_active, scale_parameter, z0)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        theta, scale, fi, hi, fi_active, scale_parameter, z0 = ctx.saved_tensors
        delta_theta = grad_output * (-2.0 * fi * hi) * fi_active
        grad_x = delta_theta / scale
        grad_scale = None
        if ctx.needs_input_grad[1]:
            gate = 1.0 - scale / 2.0 if ctx.learnable_S else -1.0 / scale
            grad_scale = _scalar_parameter_gradient(-(gate * delta_theta * theta), scale_parameter)
        grad_z0 = None
        if ctx.needs_input_grad[2]:
            grad_z0 = _scalar_parameter_gradient(-(delta_theta / scale), z0)
        return grad_x, grad_scale, grad_z0, None


class Fi(CenteredGnosticActivation):
    """Model concept fidelity with a learnable bell-shaped gnostic response.

    ``Fi`` is MAGNET's primary fidelity activation. It maps an input into the
    centered coordinate ``θ = (x - z0) / S`` and evaluates
    ``fi = sech(2θ)``, producing a bell-shaped response that peaks at ``1``
    when the input matches the learned concept center ``z0`` and decays toward
    ``0`` as the deviation grows. The learnable scale lets the activation tune
    how tightly it concentrates features around the concept manifold.

    Parameters
    ----------
    learnable_S : bool, optional
        If ``True``, optimize the scale through the gated raw parameter
        ``S_raw`` with ``S = 2σ(S_raw)`` so the effective scale stays strictly
        positive during concept learning.
    learnable_z0 : bool, optional
        If ``True``, optimize the concept center ``z0`` so the activation can
        align itself with the feature location that best represents fidelity.
    initial_S : float, optional
        Initial positive scale that determines the starting concentration width
        before training refines the concept neighborhood.
    initial_z0 : float or None, optional
        Initial concept center around which fidelity is measured. If ``None``,
        z0 is initialized to the median of input values on the first forward
        pass, providing a data-driven starting point.
    name : str or None, optional
        Optional display name used in summaries, logging, and parameter
        tracking histories.
    verbose : bool, optional
        If ``True``, emit verbose diagnostic logging for activation behavior.

    Attributes
    ----------
    S : Tensor
        Stored raw scale parameter. When the scale is learnable, this tensor is
        unconstrained and mapped to the effective positive scale in the forward
        pass.
    z0 : Tensor
        Learnable concept center used to translate the input before computing
        the gnostic coordinate.
    theta : Tensor
        Most recently computed normalized deviation, cached for inspection.
    last_output : Tensor
        Most recently computed fidelity response.

    Notes
    -----
    Gnostic Concept:
        ``Fi`` measures how strongly an input belongs to the target concept.
        High values mean concentrated feature agreement; low values mean the
        feature lies outside the learned concept neighborhood.

    Use Cases:
        Use ``Fi`` in hidden or output layers when machine-condition concepts
        should be represented by strong centered fidelity rather than unbounded
        linear activations.

    Examples
    --------
    >>> import numpy as np
    >>> from machinegnostics.magnet import Fi
    >>> activation = Fi(initial_S=1.0, initial_z0=0.0)
    >>> activation(np.array([[0.0, 0.5]])).data.shape
    (1, 2)
    """

    def __init__(
        self,
        learnable_S: bool = True,
        learnable_z0: bool = True,
        initial_S: float = 1.0,
        initial_z0: float | None = None,
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
        """Transform inputs into estimating fidelity values.

        Parameters
        ----------
        x : array-like or Tensor
            Input features whose centered deviation from ``z0`` should be
            converted into concept fidelity.
        training : bool, optional
            Present for API symmetry with other MAGNET layers. ``Fi`` uses the
            same deterministic computation in training and inference modes.

        Returns
        -------
        Tensor
            Fidelity response in the interval ``(0, 1]`` for finite inputs.

        Notes
        -----
        The layer evaluates ``fi = sech(2θ)`` with ``θ = (x - z0) / S``. If
        ``S`` is learnable, the stored raw parameter is gated as
        ``S = 2σ(S_raw)`` before the characteristic is evaluated so the scale
        stays positive throughout optimization.
        """
        # Initialize z0 from data on first forward pass if not user-provided
        self._initialize_z0_from_data(x)
        
        x = self._as_tensor(x)
        output = FiFunction.apply(x._tensor, self.S._tensor, self.z0._tensor, self.learnable_S)
        with torch.no_grad():
            scale = _effective_scale(self.S._tensor, self.learnable_S)
            theta = (x._tensor - self.z0._tensor) / scale
        self.theta = Tensor.from_torch(theta.detach())
        self.last_output = Tensor.from_torch(output)
        return Tensor.from_torch(output)
