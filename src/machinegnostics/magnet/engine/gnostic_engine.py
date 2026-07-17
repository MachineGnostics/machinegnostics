'''
ManGo - Machine Gnostics Library
Copyright (C) 2026 Nirmal Parmar

NOTE: this code is extension of src/machinegnostics/magcal/mg_weights.py. This duplicate code is for internal use in magnet engine and is not intended for external use. It is kept separate for better modularity and to avoid circular imports.

Author: Nirmal Parmar
'''

import gc

import numpy as np
from machinegnostics.magcal import GnosticsCharacteristics, ScaleParam
import logging
from machinegnostics.magcal.util.logging import get_logger

class GnosticEngine:
    '''
    Calculates Machine Gnostics weights as per different requirements.

    For internal use only.
    '''
    def __init__(self, verbose: bool = False):
        self.logger = get_logger('GnosticsWeights', level=logging.WARNING if not verbose else logging.INFO)
        self.logger.info("GnosticsWeights initialized.")

    def _get_gnostic_weights(self, z, scale_param='auto'):
        """Compute gnostic weights."""
        if scale_param == 'auto':
            self.logger.info("Computing gnostic weights and optimizing local scale...")
            z0 = np.median(z)
            zz = z / z0
            self.gc = GnosticsCharacteristics(R=zz)
            q, q1 = self.gc._get_q_q1(S=1)
            fi = self.gc._fi(q, q1)
            scale = ScaleParam()
            self.s = scale._gscale_loc(np.mean(fi))
            self.q, self.q1 = self.gc._get_q_q1(S=self.s)
            self.fi = self.gc._fi(self.q, self.q1)
            wt = self.fi**2
            self.logger.info("Gnostic weights computation complete.")
            return wt
        else:
            self.s = scale_param
            self.logger.info("Computing gnostic weights with given scale parameter...")
            z0 = np.median(z)
            zz = z / z0
            self.gc = GnosticsCharacteristics(R=zz)
            self.q, self.q1 = self.gc._get_q_q1(S=self.s)
            self.fi = self.gc._fi(self.q, self.q1)
            wt = self.fi**2
            self.logger.info("Gnostic weights computation with given scale parameter complete.")
            return wt
    
    def _get_activation(self, z, scale_param='auto'):
        """Compute gnostic activation."""
        if scale_param == 'auto':
            self.logger.info("Computing gnostic activation and optimizing local scale...")
            z0 = 1
            zz = z / z0
            self.gc = GnosticsCharacteristics(R=zz)
            q, q1 = self.gc._get_q_q1(S=1)
            fi = self.gc._fi(q, q1)
            scale = ScaleParam()
            self.s = scale._gscale_loc(np.mean(fi))
            self.q, self.q1 = self.gc._get_q_q1(S=self.s)
            self.fi = self.gc._fi(self.q, self.q1)
            activation = self.fi
            self.logger.info("Gnostic activation computation complete.")
            return activation
        else:
            self.s = scale_param
            self.logger.info("Computing gnostic activation with given scale parameter...")
            z0 = 1
            zz = z / z0
            self.gc = GnosticsCharacteristics(R=zz)
            self.q, self.q1 = self.gc._get_q_q1(S=self.s)
            self.fi = self.gc._fi(self.q, self.q1)
            activation = self.fi
            self.logger.info("Gnostic activation computation with given scale parameter complete.")
            return activation
    
    def _get_fi(self):
        return self.fi
    
    def _get_hi(self):
        self.hi = self.gc._hi(self.q, self.q1)
        return self.hi
    
    def _get_fj(self):
        self.fj = self.gc._fj(self.q, self.q1)
        return self.fj
    
    def _get_hj(self):
        self.hj = self.gc._hj(self.q, self.q1)
        return self.hj
    
    def _get_re(self):
        self.re = self.gc._rentropy(self._get_fi(), self._get_fj())
        return self.re
    
    def _get_S_local(self):
        return self.s