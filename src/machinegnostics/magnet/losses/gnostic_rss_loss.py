"""RSSLoss for MAGNET.

Developer note
--------------
Author: Nirmal Parmar, OSS Machine Gnostics

This module implements the residual-sum-of-squares variant built on the
shared gnostic loss bridge.
"""

from __future__ import annotations

import numpy as np

from machinegnostics.magcal import GnosticsCharacteristics, ScaleParam

from ._shared import Loss, Tensor, logger, _prepare_tensors, _scalar_gnostic_loss, _BaseGnosticCharc
class RSSLoss(Loss, _BaseGnosticCharc):
    """Gnostic relevance squared-sum loss.

    RSS stands for relevance squared sum. It aggregates the squared relevance
    characteristic across the batch, so it behaves like a relevance-focused
    alternative to standard residual penalties.

    Use it when the relevance signal itself is the quantity you want to drive
    down during training.
    """

    def __init__(self, S: float | str = 1, verbose: bool = False):
        Loss.__init__(self, verbose=verbose)
        _BaseGnosticCharc.__init__(self, S=S)

    def forward(self, y_pred, y_true):
        """Compute the relevance-based gnostic loss."""
        y_pred, y_true = _prepare_tensors(y_pred, y_true)
        self.y_pred, self.y_true = y_pred, y_true
        if self.verbose:
            self.logger.info("Computing rss loss for shape %s.", y_pred.data.shape)
        hi = self._get_hi(y_pred.data, y_true.data)
        value = np.sum(hi**2)
        gradient =  (4.0 / self.S_local) * (1 - hi**2) * hi
        return _scalar_gnostic_loss(y_pred, value, gradient)

    def backward(self):
        raise NotImplementedError("RSSLoss uses tensor autograd; call loss.backward() instead")
