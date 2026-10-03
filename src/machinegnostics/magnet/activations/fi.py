"""Fi activation for MAGNET.

This module contains the dedicated implementation of the estimating
fidelity characteristic used by MAGNET's gnostic activation family.
``Fi`` learns a center ``z0`` and, optionally, a scale ``S`` so the
activation can adapt to the residual geometry seen during training.
"""

from __future__ import annotations

import numpy as np
import torch

from ..core.tensor import Tensor
from ._centered import CenteredGnosticActivation, EPS


def fi(x, S: float = 1.0, z0: float = 0.0) -> np.ndarray:
    """Evaluate the estimating fidelity characteristic ``sech(2θ)``.

    The helper mirrors the layer mathematics in plain NumPy form, which
    is useful for analysis, tests, and quick experimentation outside a
    full model graph.

    Parameters
    ----------
    x : array-like
        Input values or residuals to transform.
    S : float, optional
        Positive scale controlling how quickly fidelity decays away from
        the center.
    z0 : float, optional
        Center value representing perfect alignment.

    Returns
    -------
    numpy.ndarray
        Fidelity values in the interval ``(0, 1]``.
    """
    array = np.asarray(x, dtype=np.float64)
    theta = (array - z0) / max(abs(float(S)), EPS)
    two_theta = np.clip(2.0 * theta, -30.0, 30.0)
    return np.clip(1.0 / np.cosh(two_theta), EPS, 1.0)


class Fi(CenteredGnosticActivation):
    """Learnable estimating fidelity activation ``sech(2θ)``.

    ``Fi`` is the core gnostic fidelity response. It peaks when the
    input matches the learned center ``z0`` and decays symmetrically as
    the normalized deviation ``θ = (x - z0) / S`` grows in magnitude.
    In MAGNET models, this makes ``Fi`` a natural output or hidden-layer
    activation when closeness to a concept center should be expressed as
    bounded confidence.

    Together with :class:`machinegnostics.magnet.activations.hi.Hi`, the
    activation satisfies the conservation identity ``fi² + hi² = 1``.

    Attributes
    ----------
    S : Tensor
        Trainable or fixed positive scale parameter.
    z0 : Tensor
        Trainable or fixed center parameter.
    theta : Tensor
        Cached normalized deviation from the last forward pass.
    last_output : Tensor
        Cached activation output from the last forward pass.

    Raises
    ------
    ValueError
        If ``initial_S`` is not strictly positive.

    Examples
    --------
    >>> import numpy as np
    >>> from machinegnostics.magnet import Fi
    >>> layer = Fi(learnable_S=False, learnable_z0=False, initial_S=1.0)
    >>> layer(np.array([0.0, 1.0])).shape
    (2,)
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
        """Initialize the Fi activation layer."""
        super().__init__(
            learnable_S=learnable_S,
            learnable_z0=learnable_z0,
            initial_S=initial_S,
            initial_z0=initial_z0,
            name=name,
            verbose=verbose,
        )

    def forward(self, x, training: bool = True) -> Tensor:
        """Transform inputs into fidelity values.

        Parameters
        ----------
        x : Tensor or array-like
            Input values or residuals to transform.
        training : bool, optional
            Compatibility flag for the MAGNET layer API.

        Returns
        -------
        Tensor
            Tensor with the same shape as ``x`` and values in ``(0, 1]``.

        Notes
        -----
        Gradients for ``S`` and ``z0`` are handled by PyTorch autograd
        through the wrapped torch computation graph.
        """
        x, _, two_theta = self._theta(x)
        output = torch.clamp(1.0 / torch.cosh(two_theta), min=EPS, max=1.0)
        self.last_output = Tensor.from_torch(output)
        return Tensor.from_torch(output)
