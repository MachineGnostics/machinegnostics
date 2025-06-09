'''
ManGo - Machine Gnostics Library
Copyright (C) 2025  ManGo Team

This work is licensed under the terms of the GNU General Public License version 3.0.
For more details, see <https://www.gnu.org/licenses/gpl-3.0.html>.
'''

import numpy as np
from machinegnostics.magcal import GnosticsCharacteristics, ScaleParam

class GnosticsWeights:
    '''
    Calculates Machine Gnostics weights as per different requirements.

    For internal use only.
    '''
    def _get_gnostic_weights(self, z):
        """Compute gnostic weights."""
        z0 = np.median(z)
        zz = z / z0
        gc = GnosticsCharacteristics(R=zz)
        q, q1 = gc._get_q_q1(S=1)
        fi = gc._fi(q, q1)
        scale = ScaleParam()
        s = scale._gscale_loc(np.mean(fi))
        q, q1 = gc._get_q_q1(S=s)
        wt = (2 / (q + q1))**2
        return wt