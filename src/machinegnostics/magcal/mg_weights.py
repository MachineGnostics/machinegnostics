'''
ManGo - Machine Gnostics Library
Copyright (C) 2025  ManGo Team

Author: Nirmal Parmar
'''

import numpy as np
from machinegnostics.magcal import GnosticsCharacteristics, ScaleParam
import logging
from machinegnostics.magcal.util.logging import get_logger

class GnosticsWeights:
    '''
    Calculates Machine Gnostics weights as per different requirements.

    For internal use only.
    '''
    def __init__(self, verbose: bool = False):
        self.logger = get_logger('GnosticsWeights', level=logging.WARNING if not verbose else logging.INFO)
        self.logger.info("GnosticsWeights initialized.")

    def _get_gnostic_weights(self, z):
        """Compute gnostic weights."""
        self.logger.info("Computing gnostic weights...")
        z0 = np.median(z)
        zz = z / z0
        gc = GnosticsCharacteristics(R=zz)
        q, q1 = gc._get_q_q1(S=1)
        fi = gc._fi(q, q1)
        scale = ScaleParam()
        s = scale._gscale_loc(np.mean(fi))
        q, q1 = gc._get_q_q1(S=s)
        wt = (2 / (q + q1))**2
        self.logger.info("Gnostic weights computation complete.")
        return wt