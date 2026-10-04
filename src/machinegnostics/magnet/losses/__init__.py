"""Loss exports for MAGNET.

This module keeps the standard regression and binary-classification
losses in a shallow namespace and re-exports the dedicated gnostic loss
implementations from their own files.

Examples
--------
>>> import numpy as np
>>> from machinegnostics.magnet.losses import MSE, FidelityLoss
>>> float(MSE()(np.array([[1.0]]), np.array([[0.0]])))
1.0
>>> isinstance(FidelityLoss(), Loss)
True
"""

from __future__ import annotations

import numpy as np

from ..core.tensor import Tensor
from .base import Loss, prepare_tensors
from .fidelity_loss import FidelityLoss
from .infidelity_loss import InfidelityLoss
from .information_loss import InformationLoss
from .iss_loss import ISSLoss
from .residual_entropy_loss import ResidualEntropyLoss
from .rss_loss import RSSLoss


class MSE(Loss):
    """Compute mean-squared error for regression targets.

    Mean-squared error penalizes larger residuals quadratically, making
    it the standard objective for continuous targets when large mistakes
    should be discouraged more strongly than small ones.

    Examples
    --------
    >>> import numpy as np
    >>> from machinegnostics.magnet.losses import MSE
    >>> float(MSE()(np.array([[2.0]]), np.array([[1.0]])))
    1.0
    """

    def forward(self, y_pred, y_true) -> Tensor:
        """Return the average squared difference.

        Parameters
        ----------
        y_pred : Tensor or array-like
            Predicted values.
        y_true : Tensor or array-like
            Target values with the same shape as ``y_pred``.

        Returns
        -------
        Tensor
            Scalar tensor containing the batch mean of squared error.

        Raises
        ------
        ValueError
            If prediction and target shapes differ.
        """
        y_pred, y_true = prepare_tensors(y_pred, y_true)
        return ((y_pred - y_true) ** 2).mean()


class BinaryCrossEntropy(Loss):
    """Compute binary cross-entropy for probability predictions.

    This loss is appropriate when the model outputs Bernoulli
    probabilities, usually after a sigmoid activation. It penalizes
    overconfident incorrect predictions heavily while rewarding accurate,
    calibrated probabilities.

    Examples
    --------
    >>> import numpy as np
    >>> from machinegnostics.magnet.losses import BinaryCrossEntropy
    >>> loss = BinaryCrossEntropy()
    >>> round(float(loss(np.array([[0.9]]), np.array([[1.0]]))), 3)
    0.105
    """

    def forward(self, y_pred, y_true, eps: float = 1e-12) -> Tensor:
        """Return binary cross-entropy between predictions and targets.

        Parameters
        ----------
        y_pred : Tensor or array-like
            Predicted probabilities in ``[0, 1]``.
        y_true : Tensor or array-like
            Binary targets with the same shape as ``y_pred``.
        eps : float, optional
            Numerical stability floor applied before taking logarithms.

        Returns
        -------
        Tensor
            Scalar tensor containing mean binary cross-entropy.

        Raises
        ------
        ValueError
            If prediction and target shapes differ.
        """
        y_pred, y_true = prepare_tensors(y_pred, y_true)
        clipped = y_pred.clip(eps, 1.0 - eps)
        return -(y_true * clipped.log() + (1.0 - y_true) * (1.0 - clipped).log()).mean()


def fidelity_loss(y_pred, y_true, S: float | str = 1):
    """Compute fidelity loss with the dedicated autograd-based class."""
    return FidelityLoss(S=S)(y_pred, y_true)


def infidelity_loss(y_pred, y_true, S: float | str = 1):
    """Compute infidelity loss with the dedicated autograd-based class."""
    return InfidelityLoss(S=S)(y_pred, y_true)


def irrelevance_loss(y_pred, y_true, S: float | str = 1):
    """Compute residual-entropy loss for legacy helper workflows."""
    return ResidualEntropyLoss(S=S)(y_pred, y_true)


def relevance_loss(y_pred, y_true, S: float | str = 1):
    """Compute information loss for legacy helper workflows."""
    return InformationLoss(S=S)(y_pred, y_true)


def gnostic_characteristic_loss(y_pred, y_true, S: float | str = 1):
    """Alias residual-entropy loss for compatibility with older helpers."""
    return ResidualEntropyLoss(S=S)(y_pred, y_true)


def gnostic_weighted_mse(y_pred, y_true):
    """Return standard mean-squared error for compatibility callers."""
    return MSE()(y_pred, y_true)


def gnostic_weighted_rmse(y_pred, y_true):
    """Return root mean-squared error for compatibility callers."""
    value = MSE()(y_pred, y_true)
    return value ** 0.5 if isinstance(value, Tensor) else np.sqrt(value)


def get_loss(loss):
    """Resolve a string or object into a MAGNET loss instance.

    Parameters
    ----------
    loss : None, str, Loss, or callable
        Loss specification. Strings are resolved against the standard and
        dedicated gnostic loss registry.

    Returns
    -------
    Loss or callable
        Resolved loss object. ``None`` yields :class:`MSE`.

    Raises
    ------
    ValueError
        If the string name is unknown.
    TypeError
        If the specification cannot be interpreted.
    """
    if loss is None:
        return MSE()
    if isinstance(loss, Loss) or (hasattr(loss, 'forward') and hasattr(loss, '__call__')):
        return loss
    if isinstance(loss, str):
        registry = {
            'mse': MSE(),
            'binarycrossentropy': BinaryCrossEntropy(),
            'bce': BinaryCrossEntropy(),
            'fidelity': FidelityLoss(),
            'fidelityloss': FidelityLoss(),
            'infidelity': InfidelityLoss(),
            'infidelityloss': InfidelityLoss(),
            'rss': RSSLoss(),
            'rssloss': RSSLoss(),
            'iss': ISSLoss(),
            'issloss': ISSLoss(),
            'residualentropy': ResidualEntropyLoss(),
            'residualentropyloss': ResidualEntropyLoss(),
            'information': InformationLoss(),
            'informationloss': InformationLoss(),
        }
        key = loss.replace('_', '').replace('-', '').lower()
        try:
            return registry[key]
        except KeyError as exc:
            raise ValueError(f'Unknown loss: {loss}') from exc
    raise TypeError(f'Unsupported loss specification: {type(loss)!r}')


__all__ = [
    'Loss',
    'MSE',
    'BinaryCrossEntropy',
    'FidelityLoss',
    'InfidelityLoss',
    'RSSLoss',
    'ISSLoss',
    'ResidualEntropyLoss',
    'InformationLoss',
    'fidelity_loss',
    'infidelity_loss',
    'irrelevance_loss',
    'relevance_loss',
    'gnostic_characteristic_loss',
    'gnostic_weighted_mse',
    'gnostic_weighted_rmse',
    'get_loss',
]
