"""
Gnostic-aware loss functions for neural networks.
Computes mg_loss (hi/hj), entropy, and supports scale S optimization.
"""
import numpy as np
import logging
from typing import Tuple
from machinegnostics.magcal import (GnosticsCharacteristics, DataConversion, ScaleParam)
from machinegnostics.magcal.util.logging import get_logger
from machinegnostics.magcal.util.min_max_float import np_eps_float

class GnosticLoss:
    def __init__(self, mg_loss: str = 'hi', data_form: str = 'a', verbose: bool = False):
        if mg_loss not in ('hi', 'hj'):
            raise ValueError("mg_loss must be 'hi' or 'hj'")
        if data_form not in ('a', 'm'):
            raise ValueError("data_form must be 'a' or 'm'")
        self.mg_loss = mg_loss
        self.data_form = data_form
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if verbose else logging.WARNING)
        self.verbose = verbose

    def _convert(self, arr: np.ndarray) -> np.ndarray:
        dc = DataConversion()
        if self.data_form == 'a':
            return dc._convert_az(arr)
        else:
            return dc._convert_mz(arr)

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, scale):
        # residual and converted forms
        r = y_pred - y_true
        z = self._convert(r)
        gc = GnosticsCharacteristics(R=z)
        # z_true = self._convert(y_true)
        # z_pred = self._convert(y_pred)
        # zz = np.divide(z_pred, z_true, out=np.zeros_like(z_pred), where=z_true!=0)
        # scale
        if scale == 'auto':
            q, q1 = gc._get_q_q1(S=1)
            fi = gc._fi(q, q1)
            sp = ScaleParam()
            s = sp._gscale_loc(np.mean(fi))
        else:
            s = scale
        self.S = s
        # safe ratio
        # eps = np_eps_float()
        # z_pred_safe = np.where(np.abs(z_pred) < eps, eps, z_pred)
        # zz = z_pred_safe / z_true
        # gc = GnosticsCharacteristics(zz, verbose=self.verbose)
        q, q1 = gc._get_q_q1(S=s)
        if self.mg_loss == 'hi':
            hi = gc._hi(q, q1)
            fi = gc._fi(q, q1)
            fj = gc._fj(q, q1)
            re = gc._rentropy(fi, fj)
            H = np.sum(hi ** 2)
        else:
            hj = gc._hj(q, q1)
            fi = gc._fi(q, q1)
            fj = gc._fj(q, q1)
            re = gc._rentropy(fi, fj)
            H = np.sum(hj ** 2)
        # normalized entropy
        re_norm = (re - np.min(re)) / (np.max(re) - np.min(re)) if np.max(re) != np.min(re) else re
        return H, float(np.mean(re_norm))
