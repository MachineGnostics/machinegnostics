"""Mean-squared error loss for MAGNET.

Developer note
--------------
Author: Nirmal Parmar, OSS Machine Gnostics

This module implements the standard regression MSE objective used by the
public MAGNET API and serves as the default loss when no explicit loss is
provided.
"""

from __future__ import annotations

import numpy as np

from machinegnostics.magcal import GnosticsCharacteristics, ScaleParam

from ._shared import Loss, Tensor, logger, _prepare_tensors, _scalar_gnostic_loss, _BaseGnosticCharc
class MSE(Loss):
    """Mean-squared error loss.

    This is the standard choice for regression. It penalizes large prediction
    errors more strongly than small ones by squaring the residuals:

    .. math::

       	ext{MSE} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2

    Use it when the target is continuous and you want the model to fit the
    average numerical value as closely as possible.

    Examples
    --------
    >>> import numpy as np
    >>> from machinegnostics.magnet import MSE
    >>> round(float(MSE()(np.array([[1.0]]), np.array([[0.0]]))), 3)
    1.0
    """

    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)

    def forward(self, y_pred, y_true):
        """Compute the average squared error."""
        y_pred, y_true = _prepare_tensors(y_pred, y_true)
        self.y_pred, self.y_true = y_pred, y_true
        if self.verbose:
            self.logger.info("Computing MSE for tensors with shape %s.", y_pred.data.shape)
        return ((y_pred - y_true) ** 2).mean()

    def backward(self):
        raise NotImplementedError("MSE uses tensor autograd; call loss.backward() instead")
