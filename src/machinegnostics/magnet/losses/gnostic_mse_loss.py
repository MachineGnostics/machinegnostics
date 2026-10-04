"""MSELoss for MAGNET.

Developer note
--------------
Author: Nirmal Parmar, OSS Machine Gnostics

This module maps the gnostic characteristic outputs into a mean-squared error
style objective while keeping the gradient path consistent with the shared
loss bridge.
"""

from __future__ import annotations

import numpy as np

from machinegnostics.magcal import GnosticsCharacteristics, ScaleParam

from ._shared import Loss, Tensor, logger, _prepare_tensors, _scalar_gnostic_loss, _BaseGnosticCharc
class MSELoss(Loss, _BaseGnosticCharc):
    """Gnostic weighted mean-squared error.

    This behaves like ordinary MSE, but each sample is weighted using the
    gnostic characteristic engine before averaging. In effect, the model pays
    more attention to samples that the characteristic engine considers more
    important.

    Use it when you still want a familiar squared-error objective, but with the
    gnostic weighting scheme applied to the batch.
    """

    def __init__(self, S: float | str = 1, verbose: bool = False):
        Loss.__init__(self, verbose=verbose)
        _BaseGnosticCharc.__init__(self, S=S)

    def forward(self, y_pred, y_true):
        """Compute gnostic weighted MSE."""
        y_pred, y_true = _prepare_tensors(y_pred, y_true)
        self.y_pred, self.y_true = y_pred, y_true
        if self.verbose:
            self.logger.info("Computing MSE for tensors with shape %s.", y_pred.data.shape)
        self.gw = self._get_gw(y_pred.data, y_true.data)
        value = np.mean(self.gw * (y_pred.data - y_true.data) ** 2)
        gradient = 2.0 * self.gw * (y_pred.data - y_true.data) / y_pred.data.shape[0]
        return _scalar_gnostic_loss(y_pred, value, gradient)

    def backward(self):
        raise NotImplementedError("MSELoss uses tensor autograd; call loss.backward() instead")
