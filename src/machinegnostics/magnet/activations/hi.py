"""Hi activation for MAGNET.

This module contains the dedicated implementation of MAGNET's estimating
relevance characteristic. ``Hi`` shares the same learnable center and
scale structure as ``Fi`` but maps residual geometry into a signed,
bounded response.

Author: Nirmal Parmar
Machine Gnostics (MAGNET) Implementation
"""

from __future__ import annotations

import numpy as np
import torch

from ..core.tensor import Tensor
from ._centered import EPS, CenteredGnosticActivation, _effective_scale, _scalar_parameter_gradient, _theta_components


def hi(x, S: float = 1.0, z0: float = 0.0) -> np.ndarray:
    """Evaluate the estimating relevance characteristic ``tanh(2θ)``."""
    array = np.asarray(x, dtype=np.float64)
    theta = (array - z0) / max(abs(float(S)), EPS)
    return np.tanh(np.clip(2.0 * theta, -30.0, 30.0))


class HiFunction(torch.autograd.Function):
    """Custom autograd kernel for ``hi = tanh(2θ)``.

    Mathematical Notes:
        Forward computation
        -------------------
        ``θ = (x - z0) / S`` and ``hi = tanh(2θ)``.

        Backward computation
        --------------------
        Using the conservation identity ``fi² + hi² = 1`` with
        ``fi = sech(2θ)``, the derivative is ``∂hi/∂θ = 2 fi²``.
        The custom kernel forms ``δθ = grad_output * 2 fi²`` and then applies
        ``dL/dx = δθ / S``,
        ``dL/dS_raw = -(1 - S/2) * Σ(δθ * θ)``, and
        ``dL/dz0 = -(1/S) * Σ(δθ)``.

        Numerical stability
        -------------------
        ``2θ`` is clipped to the shared safe interval. Gradients are zeroed once
        that clip saturates, which matches the clipped forward path.
    """

    @staticmethod
    def forward(ctx, x, scale_parameter, z0, learnable_S):
        scale, theta, two_theta, active = _theta_components(x, scale_parameter, z0, learnable_S)
        output = torch.tanh(two_theta)
        fi = 1.0 / torch.cosh(two_theta)
        ctx.learnable_S = learnable_S
        ctx.save_for_backward(theta, scale, fi, active, scale_parameter, z0)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        theta, scale, fi, active, scale_parameter, z0 = ctx.saved_tensors
        delta_theta = grad_output * (2.0 * fi.square()) * active
        grad_x = delta_theta / scale
        grad_scale = None
        if ctx.needs_input_grad[1]:
            gate = 1.0 - scale / 2.0 if ctx.learnable_S else -1.0 / scale
            grad_scale = _scalar_parameter_gradient(-(gate * delta_theta * theta), scale_parameter)
        grad_z0 = None
        if ctx.needs_input_grad[2]:
            grad_z0 = _scalar_parameter_gradient(-(delta_theta / scale), z0)
        return grad_x, grad_scale, grad_z0, None


class Hi(CenteredGnosticActivation):
    """Encode directional relevance around a learned gnostic concept center.

    ``Hi`` is MAGNET's signed relevance activation. It uses the same
    centered coordinate ``θ = (x - z0) / S`` as :class:`Fi`, then evaluates
    ``hi = tanh(2θ)`` to express whether the input sits below or above the
    learned concept center. The response is bounded in ``[-1, 1]`` and
    preserves directional information that is useful for residual modeling and
    concept-separation tasks.

    Parameters
    ----------
    learnable_S : bool, optional
        If ``True``, optimize the gated scale ``S = 2σ(S_raw)`` so the
        irrelevance transition width can adapt while remaining positive.
    learnable_z0 : bool, optional
        If ``True``, learn the concept center that defines zero irrelevance.
    initial_S : float, optional
        Initial transition width for the irrelevance response.
    initial_z0 : float, optional
        Initial center at which the directional response crosses zero.
    name : str or None, optional
        Optional display name used in histories and debugging output.
    verbose : bool, optional
        If ``True``, emit verbose diagnostic logging.

    Attributes
    ----------
    S : Tensor
        Stored raw scale parameter used to derive the effective positive scale.
    z0 : Tensor
        Learnable center that anchors the directional response.
    theta : Tensor
        Most recently computed centered coordinate.
    last_output : Tensor
        Most recent relevance activation values.

    Notes
    -----
    Gnostic Concept:
        ``Hi`` quantifies how directionally relevant a feature is to the
        target concept. Negative values indicate one side of the concept
        manifold, positive values indicate the other, and values near zero
        indicate concept agreement.

    Use Cases:
        ``Hi`` is well suited for residual-sensitive hidden layers, directional
        fault signatures, and complementary feature channels that should retain
        sign information.

    Examples
    --------
    >>> import numpy as np
    >>> from machinegnostics.magnet import Hi
    >>> Hi(initial_S=1.0, initial_z0=0.0)(np.array([[-1.0, 0.0, 1.0]])).data.shape
    (1, 3)
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
        """Transform inputs into signed relevance values.

        Parameters
        ----------
        x : array-like or Tensor
            Input features to map into the directional gnostic coordinate.
        training : bool, optional
            Present for layer API compatibility. The forward rule is identical
            in training and inference modes.

        Returns
        -------
        Tensor
            Signed relevance response bounded to ``[-1, 1]``.

        """
        # Initialize z0 from data on first forward pass if not user-provided
        self._initialize_z0_from_data(x)
        
        x = self._as_tensor(x)
        output = HiFunction.apply(x._tensor, self.S._tensor, self.z0._tensor, self.learnable_S)
        with torch.no_grad():
            scale = _effective_scale(self.S._tensor, self.learnable_S)
            theta = (x._tensor - self.z0._tensor) / scale
        self.theta = Tensor.from_torch(theta.detach())
        self.last_output = Tensor.from_torch(output)
        return Tensor.from_torch(output)
