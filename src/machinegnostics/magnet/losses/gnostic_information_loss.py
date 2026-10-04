"""InformationLoss for MAGNET.

Developer note
--------------
Author: Nirmal Parmar, OSS Machine Gnostics

This module measures the information characteristic of the predicted tensor
and converts it into a differentiable scalar loss through the shared gnostic
loss bridge.
"""

from __future__ import annotations

import numpy as np

from machinegnostics.magcal import GnosticsCharacteristics, ScaleParam

from ._shared import Loss, Tensor, logger, _prepare_tensors, _scalar_gnostic_loss, _BaseGnosticCharc
class InformationLoss(Loss, _BaseGnosticCharc):
    """Gnostic information loss.

    This loss converts the residual into a gnostic information quantity and
    minimizes that value. It is useful when you want the objective to reflect
    information content rather than raw distance.

    Think of it as an information-shaped regression objective rather than a
    direct error metric.
    """

    def __init__(self, S: float | str = 1, verbose: bool = False):
        Loss.__init__(self, verbose=verbose)
        _BaseGnosticCharc.__init__(self, S=S)

    def forward(self, y_pred, y_true):
        """Compute the gnostic information loss."""
        y_pred, y_true = _prepare_tensors(y_pred, y_true)
        self.y_pred, self.y_true = y_pred, y_true
        if self.verbose:
            self.logger.info("Computing information loss for shape %s.", y_pred.data.shape)
        information, fi, p = self._get_information(y_pred.data, y_true.data)
        gradient = (1 / self.S_local) * fi ** 2 * (-np.log(p) + np.log(1 - p))
        return _scalar_gnostic_loss(y_pred, np.mean(information), gradient)

    def backward(self):
        raise NotImplementedError("InformationLoss uses tensor autograd; call loss.backward() instead")
