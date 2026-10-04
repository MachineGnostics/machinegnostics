"""BinaryCrossEntropyLoss for MAGNET.

Developer note
--------------
Author: Nirmal Parmar, OSS Machine Gnostics

This module maps the gnostic characteristic pair into a probability-style
binary cross-entropy objective. It reuses the shared residual-entropy helper
path so gradients remain consistent with the rest of the gnostic losses.
"""

from __future__ import annotations

import numpy as np

from machinegnostics.magcal import GnosticsCharacteristics, ScaleParam

from ._shared import Loss, Tensor, logger, _prepare_tensors, _scalar_gnostic_loss, _BaseGnosticCharc
class BinaryCrossEntropyLoss(Loss, _BaseGnosticCharc):
    """Gnostic weighted binary cross-entropy.

    This is the gnostic analogue of binary cross-entropy. It keeps the same
    probability-based classification objective, but weights each sample with the
    gnostic characteristic engine before averaging.

    Use it for binary classification tasks when you want BCE's probabilistic
    interpretation and the gnostic weighting scheme at the same time.
    """

    def __init__(self, S: float | str = 1, verbose: bool = False):
        Loss.__init__(self, verbose=verbose)
        _BaseGnosticCharc.__init__(self, S=S)

    def forward(self, y_pred, y_true, eps=1e-12):
        """Compute gnostic weighted binary cross-entropy."""
        y_pred, y_true = _prepare_tensors(y_pred, y_true)
        self.y_pred = y_pred.clip(eps, 1 - eps)
        self.y_true = y_true
        if self.verbose:
            self.logger.info("Computing binary cross-entropy for tensors with shape %s.", y_pred.data.shape)
        self.gw = self._get_gw(y_pred.data, y_true.data)
        value = -np.mean(self.gw * (y_true.data * np.log(self.y_pred.data) + (1 - y_true.data) * np.log(1 - self.y_pred.data)))
        gradient = self.gw * (self.y_pred.data - y_true.data) / (self.y_pred.data * (1 - self.y_pred.data) * y_pred.data.shape[0])
        return _scalar_gnostic_loss(self.y_pred, value, gradient)

    def backward(self):
        raise NotImplementedError("BinaryCrossEntropyLoss uses tensor autograd; call loss.backward() instead")
