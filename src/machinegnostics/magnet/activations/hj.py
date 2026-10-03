"""Hj activation for MAGNET.

This module contains the dedicated implementation of MAGNET's
quantifying irrelevance characteristic. ``Hj`` complements ``Hi`` by
producing an unbounded signed response that grows with deviation.
"""

from __future__ import annotations

import numpy as np
import torch

from ..core.tensor import Tensor
from ._centered import EPS, MAX_MAGNITUDE, CenteredGnosticActivation, _effective_scale, _scalar_parameter_gradient, _theta_components


def hj(x, S: float = 1.0, z0: float = 0.0) -> np.ndarray:
    """Evaluate the quantifying irrelevance characteristic ``sinh(2θ)``."""
    array = np.asarray(x, dtype=np.float64)
    theta = (array - z0) / max(abs(float(S)), EPS)
    two_theta = np.clip(2.0 * theta, -30.0, 30.0)
    return np.clip(np.sinh(two_theta), -MAX_MAGNITUDE, MAX_MAGNITUDE)


class HjFunction(torch.autograd.Function):
    """Custom autograd kernel for ``hj = sinh(2θ)``.

    Mathematical Notes:
        Forward computation
        -------------------
        ``θ = (x - z0) / S`` and ``hj = sinh(2θ) = hi / fi``.

        Backward computation
        --------------------
        Since ``∂sinh(2θ)/∂θ = 2 cosh(2θ)``, ``∂hj/∂θ = 2 fj``.
        The resulting ``δθ = grad_output * 2 fj`` is propagated with the same
        MAGNET centered-coordinate rules used by the other gnostic activations.

        Numerical stability
        -------------------
        The forward path clips ``2θ`` and caps ``hj`` by ``MAX_MAGNITUDE``. The
        manual backward zeros any saturated branch so the derivative remains
        consistent with the stabilized forward pass.
    """

    @staticmethod
    def forward(ctx, x, scale_parameter, z0, learnable_S):
        scale, theta, two_theta, active = _theta_components(x, scale_parameter, z0, learnable_S)
        fj = torch.cosh(two_theta)
        hj_raw = torch.sinh(two_theta)
        output = torch.clamp(hj_raw, min=-MAX_MAGNITUDE, max=MAX_MAGNITUDE)
        hj_active = active * (torch.abs(hj_raw) < MAX_MAGNITUDE).to(dtype=x.dtype)
        ctx.learnable_S = learnable_S
        ctx.save_for_backward(theta, scale, fj, hj_active, scale_parameter, z0)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        theta, scale, fj, hj_active, scale_parameter, z0 = ctx.saved_tensors
        delta_theta = grad_output * (2.0 * fj) * hj_active
        grad_x = delta_theta / scale
        grad_scale = None
        if ctx.needs_input_grad[1]:
            gate = 1.0 - scale / 2.0 if ctx.learnable_S else -1.0 / scale
            grad_scale = _scalar_parameter_gradient(-(gate * delta_theta * theta), scale_parameter)
        grad_z0 = None
        if ctx.needs_input_grad[2]:
            grad_z0 = _scalar_parameter_gradient(-(delta_theta / scale), z0)
        return grad_x, grad_scale, grad_z0, None


class Hj(CenteredGnosticActivation):
    """Scale directional irrelevance into an unbounded gnostic response.

    ``Hj`` is MAGNET's quantifying irrelevance activation. It evaluates
    ``hj = sinh(2θ)`` with ``θ = (x - z0) / S``, which is equivalent to
    ``hj = hi / fi`` before stabilization. This gives an unbounded signed
    response that preserves direction while amplifying magnitude as the input
    moves away from the learned concept center.

    Parameters
    ----------
    learnable_S : bool, optional
        If ``True``, optimize the raw scale parameter while keeping the
        effective scale positive through ``S = 2σ(S_raw)``.
    learnable_z0 : bool, optional
        If ``True``, learn the concept center that anchors the directional
        irrelevance response.
    initial_S : float, optional
        Initial positive scale for the centered coordinate.
    initial_z0 : float or None, optional
        Initial concept center around which the gnostic response is measured. If ``None``,
        z0 is initialized to the median of input values on the first forward pass,
        providing a data-driven starting point.
    name : str or None, optional
        Optional display name used in histories and debugging output.
    verbose : bool, optional
        If ``True``, enable verbose logging for activation inspection.

    Attributes
    ----------
    S : Tensor
        Stored raw scale parameter used to derive the positive effective scale.
    z0 : Tensor
        Learnable center for the directional response.
    theta : Tensor
        Most recently computed centered deviation.
    last_output : Tensor
        Most recent quantifying irrelevance values.

    Notes
    -----
    Gnostic Concept:
        ``Hj`` expresses directional irrelevance with explicit scaling, making
        off-concept samples grow in magnitude instead of saturating. This can
        amplify gradients and highlight residual asymmetry.

    Relationship to ``Hi``:
        ``Hj`` keeps the sign of ``Hi`` but rescales it by inverse fidelity
        through ``hj = hi / fi``.

    Use Cases:
        Use ``Hj`` in diagnostic branches that need strong signed separation or
        gradient amplification for off-concept behavior.

    Examples
    --------
    >>> import numpy as np
    >>> from machinegnostics.magnet import Hj
    >>> Hj(initial_S=1.0, initial_z0=0.0)(np.array([[-1.0, 0.0, 1.0]])).data.shape
    (1, 3)
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
        """Transform inputs into quantifying irrelevance values.

        Parameters
        ----------
        x : array-like or Tensor
            Input features to map into the unbounded signed gnostic response.
        training : bool, optional
            Present for compatibility with the MAGNET layer interface.

        Returns
        -------
        Tensor
            Signed response capped to ``[-MAX_MAGNITUDE, MAX_MAGNITUDE]`` for
            numerical stability.

        Notes
        -----
        The layer evaluates ``sinh(2θ)`` and uses the exact derivative
        ``∂hj/∂θ = 2 cosh(2θ) = 2 fj`` in its custom backward. Learnable
        scales use the raw-parameter gradient derived from ``S = 2σ(S_raw)``.
        """
        # Initialize z0 from data on first forward pass if not user-provided
        self._initialize_z0_from_data(x)
        
        x = self._as_tensor(x)
        output = HjFunction.apply(x._tensor, self.S._tensor, self.z0._tensor, self.learnable_S)
        with torch.no_grad():
            scale = _effective_scale(self.S._tensor, self.learnable_S)
            theta = (x._tensor - self.z0._tensor) / scale
        self.theta = Tensor.from_torch(theta.detach())
        self.last_output = Tensor.from_torch(output)
        return Tensor.from_torch(output)
