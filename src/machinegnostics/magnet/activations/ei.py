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
    """Measure gnostic entropy relative to a learned concept center.

    ``Ei`` exposes MAGNET's entropy-style activation family. In estimating mode
    (``case='i'``) it evaluates ``ei = 1 - fi``, turning high fidelity into
    low entropy and low fidelity into high entropy. In quantifying mode
    (``case='j'``) it evaluates ``ei = fj - 1`` to preserve the historical
    unbounded residual form. The activation therefore links fidelity and
    information-style measures in one learnable centered layer.

    Parameters
    ----------
    learnable_S : bool, optional
        If ``True``, optimize the gated positive scale ``S = 2σ(S_raw)`` that
        controls how quickly entropy grows away from the concept center.
    learnable_z0 : bool, optional
        If ``True``, learn the concept center relative to which entropy is
        measured.
    initial_S : float, optional
        Initial positive scale for the centered coordinate.
    initial_z0 : float or None, optional
        Initial concept center around which the gnostic response is measured. If ``None``,
        z0 is initialized to the median of input values on the first forward pass,
        providing a data-driven starting point.
    case : {'i', 'j'}, optional
        Entropy variant to compute. ``'i'`` returns ``1 - fi`` and ``'j'``
        returns ``fj - 1``.
    name : str or None, optional
        Optional display name for diagnostics and history plots.
    verbose : bool, optional
        If ``True``, enable verbose logging for activation inspection.

    Attributes
    ----------
    S : Tensor
        Stored raw scale parameter used to derive the effective positive scale.
    z0 : Tensor
        Learnable center used to define the gnostic coordinate.
    case : str
        Active entropy formulation.
    theta : Tensor
        Most recently computed centered deviation.
    last_output : Tensor
        Most recent entropy-like response.

    Notes
    -----
    Gnostic Concept:
        Estimating-mode ``Ei`` turns fidelity concentration into an entropy
        measure, so it is useful for information-style objectives and plots.

    Relationship to FidelityLoss:
        In estimating mode, ``Ei`` matches the ``1 - fi`` quantity often used
        to visualize residual entropy alongside fidelity-driven losses.

    Use Cases:
        Use ``Ei`` when monitoring information growth, building entropy-aware
        features, or comparing fidelity and residual uncertainty directly.

    Examples
    --------
    >>> import numpy as np
    >>> from machinegnostics.magnet import Ei
    >>> Ei(case='i', initial_S=1.0)(np.array([[0.0, 1.0]])).data.shape
    (1, 2)
    """

    def __init__(
        self,
        learnable_S: bool = True,
        learnable_z0: bool = True,
        initial_S: float = 1.0,
        initial_z0: float | None = None,
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

        Parameters
        ----------
        x : array-like or Tensor
            Input features to map into the selected entropy characteristic.
        training : bool, optional
            Present for compatibility with the MAGNET layer interface.

        Returns
        -------
        Tensor
            Estimating-mode output in ``[0, 1)`` or quantifying-mode output in
            ``[0, MAX_MAGNITUDE]`` after stabilization.

        Notes
        -----
        Case ``'i'`` implements ``ei = 1 - fi`` with derivative
        ``∂ei/∂θ = 2 fi hi``. Case ``'j'`` preserves the quantifying form
        ``ei = fj - 1`` with derivative ``∂ei/∂θ = 2 fj hi``. Both use the
        same sigmoid-gated scale and exact centered-coordinate backward rules.
        """
        # Initialize z0 from data on first forward pass if not user-provided
        self._initialize_z0_from_data(x)
        
        x = self._as_tensor(x)
        output = EiFunction.apply(x._tensor, self.S._tensor, self.z0._tensor, self.learnable_S, self.case)
        with torch.no_grad():
            scale = _effective_scale(self.S._tensor, self.learnable_S)
            theta = (x._tensor - self.z0._tensor) / scale
        self.theta = Tensor.from_torch(theta.detach())
        self.last_output = Tensor.from_torch(output)
        return Tensor.from_torch(output)
