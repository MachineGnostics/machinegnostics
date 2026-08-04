"""Loss functions for MAGNET (Machine Gnostics Neural Networks).

Developer note
-------------
Author: Nirmal Parmar

This module implements standard losses and gnostic losses that reuse the
Machine Gnostics characteristic engine.

The public API is intentionally small:

- use ``MSE`` and ``BinaryCrossEntropy`` for standard regression and binary
  classification;
- use the ``Gnostic*`` losses when you want to weight the objective through
  the gnostic characteristic engine;
- use ``get_loss`` when you want string-based resolution inside ``compile``.

Examples
--------
>>> import numpy as np
>>> from  machinegnostics.magnet import BinaryCrossEntropy
>>> loss = BinaryCrossEntropy()
>>> round(float(loss(np.array([[0.9]]), np.array([[1.0]]))), 3)
0.105
"""

from __future__ import annotations

import logging

from machinegnostics.magnet.activations import hi
import numpy as np

from machinegnostics.magcal import GnosticsCharacteristics, ScaleParam

from ..core._gnostic import custom_tensor
from ..core.tensor import Tensor
from ..utils.logging import get_logger

logger = get_logger(__name__)


class Loss:
    """Base class for all MAGNET losses.

    A loss converts model predictions and targets into a single scalar that
    training tries to minimize. Concrete subclasses implement the actual
    formula, while this base class provides the shared callable interface and
    optional logging.

    Use this class as the reference point when reading the rest of the module:
    every subclass follows the same pattern of taking ``y_pred`` and ``y_true``
    and returning a scalar tensor or float.

    Parameters
    ----------
    verbose:
        If ``True``, enable info-level logging for this loss instance.

    Notes
    -----
    Subclasses should implement ``forward``. Most MAGNET losses rely on tensor
    autograd, so ``backward`` usually raises ``NotImplementedError``.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.logger = get_logger(self.__class__.__name__, logging.INFO if verbose else logging.WARNING)
        if self.verbose:
            self.logger.info("Loss initialized.")

    def forward(self, y_pred, y_true):
        """Compute a scalar loss value from predictions and targets.

        Parameters
        ----------
        y_pred:
            Model predictions.
        y_true:
            Ground-truth targets.

        Returns
        -------
        Tensor or float
            Scalar loss value.
        """
        raise NotImplementedError

    def backward(self):
        """Return the gradient of the loss with respect to predictions.

        MAGNET's tensor autograd usually handles this automatically, so
        concrete losses generally raise ``NotImplementedError`` here.
        """
        raise NotImplementedError

    def __call__(self, y_pred, y_true):
        """Shortcut for ``forward`` so losses can be called like functions."""
        return self.forward(y_pred, y_true)


def _prepare_tensors(y_pred, y_true):
    """Convert raw inputs into MAGNET tensors for loss computation."""
    y_pred = y_pred if isinstance(y_pred, Tensor) else Tensor(y_pred)
    y_true = y_true if isinstance(y_true, Tensor) else Tensor(y_true)
    logger.debug("Prepared loss tensors with shapes %s and %s.", y_pred.data.shape, y_true.data.shape)
    return y_pred, y_true


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


class _BaseGnosticCharc:
    """Shared helper for the gnostic loss family.

    The gnostic losses all start by comparing predictions to targets, then pass
    that residual through the Machine Gnostics characteristic engine. This base
    class centralizes those repeated calculations so each concrete loss can
    focus on its own objective.

    It is an internal helper and is not meant to be instantiated directly.
    """

    def __init__(self, S: float | str = 1):
        self.S = S

    def _get_fidelity(self, y_pred, y_true):
        """Compute the fidelity characteristic for a prediction residual."""
        y_diff = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
        z_y_diff = np.exp(y_diff)
        gnostic_charc = GnosticsCharacteristics(R=z_y_diff)
        if isinstance(self.S, str) and self.S == "auto":
            scale_param = ScaleParam()
            q, q1 = gnostic_charc._get_q_q1(S=1)
            fidelity = gnostic_charc._fi(q, q1)
            self.S_local = scale_param._gscale_loc(np.mean(fidelity))
            q, q1 = gnostic_charc._get_q_q1(S=self.S_local)
            fidelity = gnostic_charc._fi(q, q1)
        else:
            self.S_local = float(self.S)
            q, q1 = gnostic_charc._get_q_q1(S=self.S_local)
            fidelity = gnostic_charc._fi(q, q1)
        return fidelity

    def _get_gw(self, y_pred, y_true):
        """Compute normalized gnostic weights for a batch."""
        fi = self._get_fidelity(y_pred, y_true)
        fi2 = fi ** 2
        return fi2 / np.sum(fi2 + np.finfo(float).eps)

    def _get_rentropy(self, y_pred, y_true):
        """Compute residual entropy and the auxiliary gnostic terms."""
        y_diff = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
        z_y_diff = np.exp(y_diff)
        gnostic_charc = GnosticsCharacteristics(R=z_y_diff)
        if isinstance(self.S, str) and self.S == "auto":
            scale_param = ScaleParam()
            q, q1 = gnostic_charc._get_q_q1(S=1)
            fidelity = gnostic_charc._fi(q, q1)
            self.S_local = scale_param._gscale_loc(np.mean(fidelity))
            q, q1 = gnostic_charc._get_q_q1(S=self.S_local)
            fi = gnostic_charc._fi(q, q1)
            fj = gnostic_charc._fj(q, q1)
            hi = gnostic_charc._hi(q, q1)
            rentropy = gnostic_charc._rentropy(fi, fj)
        else:
            self.S_local = float(self.S)
            q, q1 = gnostic_charc._get_q_q1(S=self.S_local)
            fi = gnostic_charc._fi(q, q1)
            fj = gnostic_charc._fj(q, q1)
            hi = gnostic_charc._hi(q, q1)
            rentropy = gnostic_charc._rentropy(fi, fj)
        return rentropy, fi, hi

    def _get_information(self, y_pred, y_true):
        """Compute gnostic information and its auxiliary distribution terms."""
        y_diff = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
        z_y_diff = np.exp(y_diff)
        gnostic_charc = GnosticsCharacteristics(R=z_y_diff)
        if isinstance(self.S, str) and self.S == "auto":
            scale_param = ScaleParam()
            q, q1 = gnostic_charc._get_q_q1(S=1)
            fidelity = gnostic_charc._fi(q, q1)
            self.S_local = scale_param._gscale_loc(np.mean(fidelity))
            q, q1 = gnostic_charc._get_q_q1(S=self.S_local)
            fi = gnostic_charc._fi(q, q1)
            hi = gnostic_charc._hi(q, q1)
            p = gnostic_charc._idistfun(hi)
            information = gnostic_charc._info_i(p)
        else:
            self.S_local = float(self.S)
            q, q1 = gnostic_charc._get_q_q1(S=self.S_local)
            fi = gnostic_charc._fi(q, q1)
            hi = gnostic_charc._hi(q, q1)
            p = gnostic_charc._idistfun(hi)
            information = gnostic_charc._info_i(p)
        return information, fi, p

    def _get_fihi(self, y_pred, y_true):
        """Return the fidelity, irrelevance, and effective scale value."""
        y_diff = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
        z_y_diff = np.exp(y_diff)
        gnostic_charc = GnosticsCharacteristics(R=z_y_diff)
        if isinstance(self.S, str) and self.S == "auto":
            scale_param = ScaleParam()
            q, q1 = gnostic_charc._get_q_q1(S=1)
            fi = gnostic_charc._fi(q, q1)
            self.S_local = scale_param._gscale_loc(np.mean(fi))
            q, q1 = gnostic_charc._get_q_q1(S=self.S_local)
            fi = gnostic_charc._fi(q, q1)
            hi = gnostic_charc._hi(q, q1)
        else:
            self.S_local = float(self.S)
            q, q1 = gnostic_charc._get_q_q1(S=self.S_local)
            fi = gnostic_charc._fi(q, q1)
            hi = gnostic_charc._hi(q, q1)
        return fi, hi, self.S_local

    def _get_hj(self, y_pred, y_true):
        """Return the irrelevance characteristic for a prediction residual."""
        y_diff = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
        z_y_diff = np.exp(y_diff)
        gnostic_charc = GnosticsCharacteristics(R=z_y_diff)
        if isinstance(self.S, str) and self.S == "auto":
            scale_param = ScaleParam()
            q, q1 = gnostic_charc._get_q_q1(S=1)
            fi = gnostic_charc._fi(q, q1)
            self.S_local = scale_param._gscale_loc(np.mean(fi))
            q, q1 = gnostic_charc._get_q_q1(S=self.S_local)
            hj = gnostic_charc._hj(q, q1)
        else:
            self.S_local = float(self.S)
            q, q1 = gnostic_charc._get_q_q1(S=self.S_local)
            hj = gnostic_charc._hj(q, q1)
        return hj

    def _get_hi(self, y_pred, y_true):
        """Return the relevance characteristic for a prediction residual."""
        y_diff = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
        z_y_diff = np.exp(y_diff)
        gnostic_charc = GnosticsCharacteristics(R=z_y_diff)
        if isinstance(self.S, str) and self.S == "auto":
            scale_param = ScaleParam()
            q, q1 = gnostic_charc._get_q_q1(S=1)
            fi = gnostic_charc._fi(q, q1)
            self.S_local = scale_param._gscale_loc(np.mean(fi))
            q, q1 = gnostic_charc._get_q_q1(S=self.S_local)
            hi = gnostic_charc._hi(q, q1)
        else:
            self.S_local = float(self.S)
            q, q1 = gnostic_charc._get_q_q1(S=self.S_local)
            hi = gnostic_charc._hi(q, q1)
        return hi


def _scalar_gnostic_loss(y_pred, value, gradient):
    """Wrap a scalar gnostic quantity in a differentiable tensor."""
    return custom_tensor(value, y_pred, gradient)


class GnosticFidelity(Loss, _BaseGnosticCharc):
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
            self.logger.info("Computing gnostic fidelity loss for shape %s.", y_pred.data.shape)
        fi, hi, _ = self._get_fihi(y_pred.data, y_true.data)
        value = np.mean(fi)
        prime = (2 / self.S_local) * fi * hi
        return _scalar_gnostic_loss(y_pred, value, prime)

    def backward(self):
        raise NotImplementedError("GnosticFidelity uses tensor autograd; call loss.backward() instead")


class GnosticInfidelity(Loss, _BaseGnosticCharc):
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
        """Compute the infidelity-based gnostic loss."""
        y_pred, y_true = _prepare_tensors(y_pred, y_true)
        self.y_pred, self.y_true = y_pred, y_true
        if self.verbose:
            self.logger.info("Computing gnostic infidelity loss for shape %s.", y_pred.data.shape)
        fi, hi, _ = self._get_fihi(y_pred.data, y_true.data)
        value = np.mean(1.0 / (fi + np.finfo(float).eps))
        gradient = (2 / self.S_local) * (hi / (fi + np.finfo(float).eps))
        gradient = np.clip(gradient, -1e12, 1e12)
        return _scalar_gnostic_loss(y_pred, value, gradient)

    def backward(self):
        raise NotImplementedError("GnosticInfidelity uses tensor autograd; call loss.backward() instead")

class GnosticRSS(Loss, _BaseGnosticCharc):
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
            self.logger.info("Computing gnostic relevance loss for shape %s.", y_pred.data.shape)
        hi = self._get_hi(y_pred.data, y_true.data)
        value = np.sum(hi**2)
        gradient =  (4.0 / self.S_local) * (1 - hi**2) * hi
        return _scalar_gnostic_loss(y_pred, value, gradient)

    def backward(self):
        raise NotImplementedError("GnosticRSS uses tensor autograd; call loss.backward() instead")

class GnosticISS(Loss, _BaseGnosticCharc):
    """Gnostic irrelevance squared-sum loss.

    ISS stands for irrelevance squared sum. It aggregates the squared
    irrelevance characteristic across the batch, so the objective is driven by
    how strongly the model lands in the irrelevance regime.

    Use it when the irrelevance signal is the behavior you want to encourage or
    suppress explicitly.
    """

    def __init__(self, S: float | str = 1, verbose: bool = False):
        Loss.__init__(self, verbose=verbose)
        _BaseGnosticCharc.__init__(self, S=S)

    def forward(self, y_pred, y_true):
        """Compute the irrelevance-based gnostic loss."""
        y_pred, y_true = _prepare_tensors(y_pred, y_true)
        self.y_pred, self.y_true = y_pred, y_true
        if self.verbose:
            self.logger.info("Computing gnostic irrelevance loss for shape %s.", y_pred.data.shape)
        hj = self._get_hj(y_pred.data, y_true.data)
        value = np.sum(hj**2)
        gradient =  -(4.0 / self.S_local) * (1 - hj**2) * hj
        return _scalar_gnostic_loss(y_pred, value, gradient)

    def backward(self):
        raise NotImplementedError("GnosticISS uses tensor autograd; call loss.backward() instead")


class GnosticInformation(Loss, _BaseGnosticCharc):
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
            self.logger.info("Computing gnostic information loss for shape %s.", y_pred.data.shape)
        information, fi, p = self._get_information(y_pred.data, y_true.data)
        gradient = (1 / self.S_local) * fi ** 2 * (-np.log(p) + np.log(1 - p))
        return _scalar_gnostic_loss(y_pred, np.mean(information), gradient)

    def backward(self):
        raise NotImplementedError("GnosticInformation uses tensor autograd; call loss.backward() instead")


class GnosticResidualEntropy(Loss, _BaseGnosticCharc):
    """Gnostic residual entropy loss.

    This loss measures residual entropy in the gnostic characteristic space.
    It is a good choice when you want the model to minimize uncertainty in the
    residual rather than only minimizing the residual magnitude itself.

    Use it as the entropy-shaped counterpart to the standard regression losses.
    """

    def __init__(self, S: float | str = 1, verbose: bool = False):
        Loss.__init__(self, verbose=verbose)
        _BaseGnosticCharc.__init__(self, S=S)

    def forward(self, y_pred, y_true):
        """Compute the residual-entropy-based gnostic loss."""
        y_pred, y_true = _prepare_tensors(y_pred, y_true)
        self.y_pred, self.y_true = y_pred, y_true
        if self.verbose:
            self.logger.info("Computing gnostic residual-entropy loss for shape %s.", y_pred.data.shape)
        rentropy, fi, hi = self._get_rentropy(y_pred.data, y_true.data)
        gradient = (2 / self.S_local) * (-fi * hi + (hi / (fi + np.finfo(float).eps)))
        return _scalar_gnostic_loss(y_pred, np.mean(rentropy), gradient)

    def backward(self):
        raise NotImplementedError("GnosticResidualEntropy uses tensor autograd; call loss.backward() instead")


class GnosticMSE(Loss, _BaseGnosticCharc):
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
            self.logger.info("Computing gnostic MSE for tensors with shape %s.", y_pred.data.shape)
        self.gw = self._get_gw(y_pred.data, y_true.data)
        value = np.mean(self.gw * (y_pred.data - y_true.data) ** 2)
        gradient = 2.0 * self.gw * (y_pred.data - y_true.data) / y_pred.data.shape[0]
        return _scalar_gnostic_loss(y_pred, value, gradient)

    def backward(self):
        raise NotImplementedError("GnosticMSE uses tensor autograd; call loss.backward() instead")


class GnosticBinaryCrossEntropy(Loss, _BaseGnosticCharc):
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
            self.logger.info("Computing gnostic BCE for tensors with shape %s.", y_pred.data.shape)
        self.gw = self._get_gw(y_pred.data, y_true.data)
        value = -np.mean(self.gw * (y_true.data * np.log(self.y_pred.data) + (1 - y_true.data) * np.log(1 - self.y_pred.data)))
        gradient = self.gw * (self.y_pred.data - y_true.data) / (self.y_pred.data * (1 - self.y_pred.data) * y_pred.data.shape[0])
        return _scalar_gnostic_loss(self.y_pred, value, gradient)

    def backward(self):
        raise NotImplementedError("GnosticBinaryCrossEntropy uses tensor autograd; call loss.backward() instead")


def gnostic_weighted_mse(y_pred, y_true, S: float | str = 1):
    """Convenience wrapper for gnostic weighted MSE.

    Parameters
    ----------
    y_pred:
        Model predictions.
    y_true:
        Ground-truth targets.
    S:
        Scale parameter for the gnostic characteristic engine.
    """
    logger.debug("Calling gnostic_weighted_mse with scale %s.", S)
    return GnosticMSE(S=S)(y_pred, y_true)


def gnostic_weighted_rmse(y_pred, y_true, S: float | str = 1):
    """Convenience wrapper for gnostic weighted RMSE."""
    value = GnosticMSE(S=S)(y_pred, y_true)
    return value ** 0.5 if isinstance(value, Tensor) else np.sqrt(value)


def fidelity_loss(y_pred, y_true, S: float | str = 1):
    """Convenience wrapper for the fidelity-based gnostic loss."""
    return GnosticFidelity(S=S)(y_pred, y_true)


def infidelity_loss(y_pred, y_true, S: float | str = 1):
    """Convenience wrapper for the infidelity-based gnostic loss."""
    return GnosticInfidelity(S=S)(y_pred, y_true)


def irrelevance_loss(y_pred, y_true, S: float | str = 1):
    """Convenience wrapper for the residual-entropy gnostic loss."""
    return GnosticResidualEntropy(S=S)(y_pred, y_true)


def relevance_loss(y_pred, y_true, S: float | str = 1):
    """Convenience wrapper for the gnostic information loss."""
    return GnosticInformation(S=S)(y_pred, y_true)


def gnostic_characteristic_loss(y_pred, y_true, S: float | str = 1):
    """Alias for the gnostic residual-entropy loss."""
    return GnosticResidualEntropy(S=S)(y_pred, y_true)


def get_loss(loss):
    """Resolve a string or loss instance into a callable loss object.

    Parameters
    ----------
    loss:
        ``None`` for the default ``MSE`` loss, a loss instance, or a string
        such as ``"mse"``, ``"bce"``, ``"gnosticmse"``, or
        ``"gnosticbinarycrossentropy"``.

    Returns
    -------
    Loss
        Resolved loss object.

    Examples
    --------
    >>> from  machinegnostics.magnet import get_loss
    >>> get_loss("mse")
    MSE()
    """
    if loss is None:
        return MSE()
    if isinstance(loss, Loss):
        return loss
    if isinstance(loss, str):
        registry = {
            "mse": MSE(),
            "binarycrossentropy": BinaryCrossEntropy(),
            "bce": BinaryCrossEntropy(),
            "gnosticmse": GnosticMSE(),
            "gnosticbinarycrossentropy": GnosticBinaryCrossEntropy(),
            "gnosticfidelity": GnosticFidelity(),
            "gnosticinfidelity": GnosticInfidelity(),
            "gnosticinformation": GnosticInformation(),
            "gnosticresidualentropy": GnosticResidualEntropy(),
            "gnosticrss": GnosticRSS(),
            "gnosticiss": GnosticISS(),
        }
        key = loss.replace("_", "").replace("-", "").lower()
        try:
            return registry[key]
        except KeyError as exc:
            raise ValueError(f"Unknown loss: {loss}") from exc
    raise TypeError(f"Unsupported loss specification: {type(loss)!r}")
