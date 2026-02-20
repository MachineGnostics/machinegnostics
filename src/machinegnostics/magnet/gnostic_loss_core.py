"""
Core gnostic loss implementation for Magnet models.

Developer notes
---------------
This module is the *single source of truth* for sample-wise gnostic weights.
The training loop should compute residual-dependent weights only through
`GnosticLoss.compute(...)` and then reuse `self.gw` downstream.

Design goals:
- Avoid duplicated gnostic-weight logic in multiple classes.
- Keep output backward compatible with existing `(H, rentropy, p, info)` contract.
- Expose computed sample weights (`gw`) and scale (`S`) for model internals.
"""

from __future__ import annotations

import logging
import numpy as np

from machinegnostics.magcal import GnosticsCharacteristics, DataConversion, ScaleParam
from machinegnostics.magcal.util.logging import get_logger


class GnosticLoss:
    """Compute gnostic loss, entropy, and sample-wise gnostic weights.

    Parameters
    ----------
    mg_loss:
        Either `'hi'` or `'hj'` for the magnet criterion variant.
    data_form:
        `'a'` (additive) or `'m'` (multiplicative) data transformation.
    verbose:
        Enables debug logging.
    gnostic_characteristics:
        If `True`, computes and returns auxiliary distribution outputs.

    Attributes after `compute`
    --------------------------
    gw:
        Normalized per-sample weights derived from residual-space gnostic statistics.
    S:
        Active gnostic scale used in the latest computation.
    """

    def __init__(self,
                 mg_loss: str = 'hi',
                 data_form: str = 'a',
                 verbose: bool = False,
                 gnostic_characteristics: bool = False):
        if mg_loss not in ('hi', 'hj'):
            raise ValueError("mg_loss must be 'hi' or 'hj'")
        if data_form not in ('a', 'm'):
            raise ValueError("data_form must be 'a' or 'm'")
        self.mg_loss = mg_loss
        self.data_form = data_form
        self.gnostic_characteristics = gnostic_characteristics
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if verbose else logging.WARNING)
        self.verbose = verbose
        self.gw = None
        self.S = None

    def _convert(self, arr: np.ndarray) -> np.ndarray:
        dc = DataConversion()
        if self.data_form == 'a':
            return dc._convert_az(arr)
        return dc._convert_mz(arr)

    @staticmethod
    def _normalize_weights(weights: np.ndarray, n_samples: int) -> np.ndarray:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.size != n_samples:
            w = np.ones(n_samples, dtype=np.float64)
        w = np.abs(w)
        s = float(np.sum(w))
        if s <= 0 or not np.isfinite(s):
            return np.ones(n_samples, dtype=np.float64) / float(n_samples)
        return w / s

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, scale):
        """Compute gnostic loss tuple and update internal gnostic weights.

        Returns
        -------
        tuple
            `(H, rentropy_mean, p, info)` for backward compatibility.
        """
        r = y_pred - y_true
        z = self._convert(r)
        gc = GnosticsCharacteristics(R=z)

        if scale == 'auto':
            q_tmp, q1_tmp = gc._get_q_q1(S=1)
            fi_tmp = gc._fi(q_tmp, q1_tmp)
            sp = ScaleParam()
            s = sp._gscale_loc(np.mean(fi_tmp))
        else:
            s = scale
        self.S = s

        q, q1 = gc._get_q_q1(S=s)
        fi = gc._fi(q, q1)
        fj = gc._fj(q, q1)
        re = gc._rentropy(fi, fj)

        if self.mg_loss == 'hi':
            criterion = gc._hi(q, q1)
        else:
            criterion = gc._hj(q, q1)

        H = float(np.sum(criterion ** 2))

        # Single source of sample-wise gnostic weights.
        # Reduce all non-sample axes and normalize.
        fi_arr = np.asarray(fi)
        if fi_arr.ndim == 1:
            gw_raw = np.square(fi_arr)
        else:
            reduce_axes = tuple(range(fi_arr.ndim - 1))
            gw_raw = np.square(np.mean(fi_arr, axis=reduce_axes))
        self.gw = self._normalize_weights(gw_raw, n_samples=r.shape[0])

        if self.gnostic_characteristics:
            p = gc._idistfun(criterion)
            info = gc._info_i(p)
        else:
            p = info = None

        re_min = np.min(re)
        re_max = np.max(re)
        if re_max != re_min:
            re_norm = (re - re_min) / (re_max - re_min)
        else:
            re_norm = re

        return H, float(np.mean(re_norm)), p, info
