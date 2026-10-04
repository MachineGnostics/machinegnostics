"""InfidelityLoss for MAGNET.

Developer note
--------------
Author: Nirmal Parmar, OSS Machine Gnostics

This module measures the infidelity characteristic of the predicted tensor and
turns it into a differentiable scalar loss through the shared gnostic loss
bridge.
"""

from __future__ import annotations

import numpy as np

from machinegnostics.magcal import GnosticsCharacteristics, ScaleParam

from ._shared import Loss, Tensor, logger, _prepare_tensors, _scalar_gnostic_loss, _BaseGnosticCharc
class InfidelityLoss(Loss, _BaseGnosticCharc):
    """Gnostic infidelity loss.

    This is the complementary gnostic loss to :class:`GnosticFidelity`. It
    emphasizes the infidelity side of the residual structure, which can be
    useful when you want to penalize deviation through the alternative gnostic
    characteristic.

    A good mental model is: use this when fidelity-based weighting is not the
    right inductive bias and you want the complementary residual signal.
    """

    def __init__(self, S: float | str = 1, verbose: bool = False):
        Loss.__init__(self, verbose=verbose)
        _BaseGnosticCharc.__init__(self, S=S)

    def forward(self, y_pred, y_true):
        """Compute the infidelity-based gnostic loss.

        This mirrors the working NumPy notebook implementation:

        .. math::

            z = (\hat{y} - y) / S
            q = e^z
            fi = 2 / (q^2 + q^{-2})
            L = mean(1 / fi)

        The returned tensor carries a custom gradient for ``y_pred`` so the
        standard MAGNET autograd flow can still train the model.
        """
        y_pred, y_true = _prepare_tensors(y_pred, y_true)
        self.y_pred, self.y_true = y_pred, y_true
        if self.verbose:
            self.logger.info("Computing infidelity loss for shape %s.", y_pred.data.shape)

        residual = np.asarray(y_pred.data, dtype=np.float64) - np.asarray(y_true.data, dtype=np.float64)

        if isinstance(self.S, str) and self.S == "auto":
            q_seed = np.exp(np.clip(residual, -20.0, 20.0))
            fi_seed = 2.0 / (q_seed**2 + q_seed**-2)
            self.S_local = max(ScaleParam()._gscale_loc(np.mean(fi_seed)), 0.01)
        else:
            self.S_local = float(self.S)

        z = residual / self.S_local
        q = np.exp(np.clip(z, -20.0, 20.0))
        fi = 2.0 / (q**2 + q**-2)
        value = np.mean(1.0 / (fi + np.finfo(float).eps))
        h = np.tanh(2.0 * z)
        gradient = ((2.0 / self.S_local) * (h / (fi + np.finfo(float).eps))) / y_pred.data.shape[0]
        return _scalar_gnostic_loss(y_pred, value, gradient)

    def backward(self):
        raise NotImplementedError("InfidelityLoss uses tensor autograd; call loss.backward() instead")
