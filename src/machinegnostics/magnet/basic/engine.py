'''
ManGo - Machine Gnostics Library
Copyright (C) 2026 Nirmal Parmar

NOTE: this code is extension of src/machinegnostics/magcal/mg_weights.py. This duplicate code is for internal use in magnet engine and is not intended for external use. It is kept separate for better modularity and to avoid circular imports.

Author: Nirmal Parmar
'''

import numpy as np
from machinegnostics.magcal import GnosticsCharacteristics, ScaleParam
import logging
from machinegnostics.magcal.util.logging import get_logger

class GnosticEngine:
    '''
    Gnostic Engine Class

    Calculates Machine Gnostics weights as per different requirements.

    For internal use only.
    '''
    def __init__(self,
                 S: float | str = 2.0,
                 verbose: bool = False):
        self.S = S
        self.verbose = verbose
        self.logger = get_logger(self.__class__.__name__, level=logging.WARNING if not self.verbose else logging.INFO)
        self.logger.info(f"{self.__class__.__name__} initialized.")

    def _get_gnostic_i_weights(self, grad_output, scale_param='auto'):
        """Compute gnostic weights."""
        if isinstance(scale_param, str) and scale_param == 'auto':
            z0 = np.median(grad_output)
            zz = np.exp(grad_output - z0)
            self.gc = GnosticsCharacteristics(R=zz)
            q, q1 = self.gc._get_q_q1(S=1)
            fi = self.gc._fi(q, q1)
            scale = ScaleParam()
            self.S_local = scale._gscale_loc(np.mean(fi))
            self.q, self.q1 = self.gc._get_q_q1(S=self.S_local)
            self.fi = self.gc._fi(self.q, self.q1)
            wt = self.fi**2 / (np.sum(self.fi**2) + np.finfo(float).eps)  # Normalize weights to sum to 1
            return wt
        else:
            self.S_local = scale_param
            z0 = np.median(grad_output)
            zz = np.exp(grad_output - z0)
            self.gc = GnosticsCharacteristics(R=zz)
            self.q, self.q1 = self.gc._get_q_q1(S=self.S_local)
            self.fi = self.gc._fi(self.q, self.q1)
            wt = self.fi** 2 / (np.sum(self.fi**2) + np.finfo(float).eps)  # Normalize weights to sum to 1
            return wt

    def _get_gnostic_j_weights(self, grad_output, scale_param='auto'):
        """Compute gnostic weights."""
        gw = self._get_gnostic_i_weights(grad_output, scale_param)
        gw_j = 1 / (gw + np.finfo(float).eps)  # Avoid division by zero
        return gw_j / np.sum(gw_j)  # Normalize weights to sum to 1