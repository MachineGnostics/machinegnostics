"""Binary cross-entropy loss for MAGNET.

Developer note
--------------
Author: Nirmal Parmar, OSS Machine Gnostics

This module implements the standard probability-space binary cross-entropy
objective used by MAGNET when the model already produces sigmoid-like
predictions.
"""

from __future__ import annotations

import numpy as np

from machinegnostics.magcal import GnosticsCharacteristics, ScaleParam

from ._shared import Loss, Tensor, logger, _prepare_tensors, _scalar_gnostic_loss, _BaseGnosticCharc
class BinaryCrossEntropy(Loss):
    """Binary cross-entropy loss for probability outputs.

    Use this loss for binary classification when the model output represents a
    probability, usually after a ``Sigmoid`` layer. It rewards confident
    correct predictions and heavily penalizes confident wrong ones.

    The loss is based on:

    .. math::

       -\frac{1}{n} \sum_{i=1}^{n} \left[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right]

    In practice, this is the right choice when your labels are 0/1 and you
    want calibrated probabilities rather than raw scores.

    Examples
    --------
    >>> import numpy as np
    >>> from machinegnostics.magnet import BinaryCrossEntropy
    >>> loss = BinaryCrossEntropy()
    >>> round(float(loss(np.array([[0.9]]), np.array([[1.0]]))), 3)
    0.105
    """

    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)

    def forward(self, y_pred, y_true, eps=1e-12):
        """Compute binary cross-entropy between predictions and labels."""
        y_pred, y_true = _prepare_tensors(y_pred, y_true)
        self.y_pred, self.y_true = y_pred.clip(eps, 1 - eps), y_true
        if self.verbose:
            self.logger.info("Computing binary cross-entropy for tensors with shape %s.", y_pred.data.shape)
        return -(y_true * self.y_pred.log() + (1 - y_true) * (1 - self.y_pred).log()).mean()

    def backward(self):
        raise NotImplementedError("BinaryCrossEntropy uses tensor autograd; call loss.backward() instead")
