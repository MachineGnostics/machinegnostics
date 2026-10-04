"""FidelityLoss for MAGNET.

Developer note
--------------
Author: Nirmal Parmar, OSS Machine Gnostics

This module measures the fidelity characteristic of the predicted tensor and
converts it into a differentiable scalar loss through the shared gnostic loss
bridge.
"""

from __future__ import annotations

import numpy as np

from machinegnostics.magcal import GnosticsCharacteristics, ScaleParam

from ._shared import Loss, Tensor, logger, _prepare_tensors, _scalar_gnostic_loss, _BaseGnosticCharc
class FidelityLoss(Loss, _BaseGnosticCharc):
    """Gnostic fidelity loss.

    This loss emphasizes the fidelity side of the gnostic characteristic split.
    It is useful when you want the model to minimize the residual in a way that
    follows the gnostic fidelity weighting rather than plain squared error.

    Use it when you want a regression-style objective but with the gnostic
    characteristic engine shaping the gradient.
    """

    def __init__(self, S: float | str = 1, verbose: bool = False):
        Loss.__init__(self, verbose=verbose)
        _BaseGnosticCharc.__init__(self, S=S)

    def forward(self, y_pred, y_true):
        """Compute the fidelity-based gnostic loss."""
        y_pred, y_true = _prepare_tensors(y_pred, y_true)
        self.y_pred, self.y_true = y_pred, y_true
        if self.verbose:
            self.logger.info("Computing fidelity loss for shape %s.", y_pred.data.shape)
        fi, hi, _ = self._get_fihi(y_pred.data, y_true.data)
        value = np.mean(fi)
        prime = (2 / self.S_local) * fi * hi
        return _scalar_gnostic_loss(y_pred, value, prime)

    def backward(self):
        raise NotImplementedError("FidelityLoss uses tensor autograd; call loss.backward() instead")
