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
    def __init__(self, verbose: bool = False):
        self.logger = get_logger(self.__class__.__name__, level=logging.WARNING if not verbose else logging.INFO)
        self.logger.info(f"{self.__class__.__name__} initialized.")

    def _get_gnostic_weights(self, z, scale_param='auto'):
        """Compute gnostic weights."""
        if scale_param == 'auto':
            self.logger.debug("Computing gnostic weights and optimizing local scale...")
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
            self.logger.debug("Gnostic weights computation complete.")
            return wt
        else:
            self.s = scale_param
            self.logger.debug("Computing gnostic weights with given scale parameter...")
            z0 = np.median(z)
            zz = z / z0
            self.gc = GnosticsCharacteristics(R=zz)
            self.q, self.q1 = self.gc._get_q_q1(S=self.s)
            self.fi = self.gc._fi(self.q, self.q1)
            wt = self.fi**2
            self.logger.debug("Gnostic weights computation with given scale parameter complete.")
            return wt
    
    def _get_activation(self, z_fi, scale_param='auto', activation_type='fi'):
        """Compute gnostic activation."""
        if scale_param == 'auto':
            self.logger.debug("Computing gnostic activation and optimizing local scale...")
            # z0 = np.median(z)
            zz = z_fi
            self.gc = GnosticsCharacteristics(R=zz)
            q, q1 = self.gc._get_q_q1(S=1)
            fi = self.gc._fi(q, q1)
            scale = ScaleParam()
            self.s = scale._gscale_loc(np.mean(fi))
            self.q, self.q1 = self.gc._get_q_q1(S=self.s)
            if activation_type == 'fi':
                self.acti = self.gc._fi(self.q, self.q1)
            elif activation_type == 'hi':
                self.acti = self.gc._hi(self.q, self.q1)
            elif activation_type == 'fj':
                self.acti = self.gc._fj(self.q, self.q1)
            elif activation_type == 'hj':
                self.acti = self.gc._hj(self.q, self.q1)
            else:
                raise ValueError(f"Invalid activation_type: {activation_type}. Must be one of ['fi', 'hi', 'fj', 'hj'].")
            activation = self.acti
            self.logger.debug("Gnostic activation computation complete.")
            return activation
        else:
            self.s = scale_param
            self.logger.debug("Computing gnostic activation with given scale parameter...")
            # z0 = np.median(z)
            zz = z_fi
            self.gc = GnosticsCharacteristics(R=zz)
            self.q, self.q1 = self.gc._get_q_q1(S=self.s)
            self.fi = self.gc._fi(self.q, self.q1)
            if activation_type == 'fi':
                self.acti = self.gc._fi(self.q, self.q1)
            elif activation_type == 'hi':
                self.acti = self.gc._hi(self.q, self.q1)
            elif activation_type == 'fj':
                self.acti = self.gc._fj(self.q, self.q1)
            elif activation_type == 'hj':
                self.acti = self.gc._hj(self.q, self.q1)
            else:
                raise ValueError(f"Invalid activation_type: {activation_type}. Must be one of ['fi', 'hi', 'fj', 'hj'].")
            activation = self.acti
            self.logger.debug("Gnostic activation computation with given scale parameter complete.")
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