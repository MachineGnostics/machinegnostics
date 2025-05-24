'''
ManGo - Machine Gnostics Library
Copyright (C) 2025  ManGo Team

This work is licensed under the terms of the GNU General Public License version 3.0.
For more details, see <https://www.gnu.org/licenses/gpl-3.0.html>.

ideas:
- LocS
- GlobS
- VarS
'''
import numpy as np
from machinegnostics.magcal import GnosticsCharacteristics

class ScaleParam():
    """
    A Machine Gnostic class to compute and optimize scale parameter for different gnostic distribution functions

    Parameters
    ----------
        F: Input Parameter, e.g., fidelity of the data
    """
    
    # def _gscale_loc(self, F):
    #     '''
    #     For internal use only

    #     calculates the local scale parameter for given calculated F at Scale = 1.
    #     S with be in the same shape as F.
    #     Solve for scale parameter using Newton-Raphson."
    #     '''
    #     m2pi = 2 / np.pi
    #     sqrt2 = np.sqrt(2)
        
    #     if F < m2pi * sqrt2 / 3:
    #         S = np.pi
    #     elif F < m2pi:
    #         S = 3 * np.pi / 4
    #     elif F < m2pi * sqrt2:
    #         S = np.pi / 2
    #     else:
    #         S = np.pi / 4

    #     epsilon = 1e-5
    #     for _ in range(100):
    #         delta = (np.sin(S) - S * F) / (np.cos(S) - F)
    #         S -= delta
    #         if abs(delta) < epsilon:
    #             break
    #     return S * m2pi
    
    def _gscale_loc(self, F):
        '''
        Calculates the local scale parameter for given calculated F at Scale = 1.
        Supports both scalar and array-like F.
        Uses Newton-Raphson method for each value if F is array-like.
        '''
        m2pi = 2 / np.pi
        sqrt2 = np.sqrt(2)
        epsilon = 1e-5

        def _single_scale(f):
            if f < m2pi * sqrt2 / 3:
                S = np.pi
            elif f < m2pi:
                S = 3 * np.pi / 4
            elif f < m2pi * sqrt2:
                S = np.pi / 2
            else:
                S = np.pi / 4
            for _ in range(100):
                delta = (np.sin(S) - S * f) / (np.cos(S) - f)
                S -= delta
                if abs(delta) < epsilon:
                    break
            return S * m2pi

        # Check if F is scalar
        if np.isscalar(F):
            return _single_scale(F)
        else:
            F = np.asarray(F)
            return np.array([_single_scale(f) for f in F])