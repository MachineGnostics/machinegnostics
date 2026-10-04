"""Shared loss helpers for MAGNET.

Developer note
--------------
Author: Nirmal Parmar, OSS Machine Gnostics

This module centralizes the shared scalar-loss bridge used by the gnostic
loss family. It converts prediction residuals into the underlying Machine
Gnostics characteristics and keeps the repeated fidelity, entropy, and
information calculations in one place.
"""

from __future__ import annotations

import logging

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

def _prepare_tensors(y_pred, y_true):
	y_pred = y_pred if isinstance(y_pred, Tensor) else Tensor(y_pred)
	y_true = y_true if isinstance(y_true, Tensor) else Tensor(y_true)
	logger.debug("Prepared loss tensors with shapes %s and %s.", y_pred.data.shape, y_true.data.shape)
	return y_pred, y_true


def _scalar_gnostic_loss(y_pred, value, gradient):
	return custom_tensor(value, y_pred, gradient)
