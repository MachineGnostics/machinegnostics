'''
Class for scale parameter computation and optimization for given data

ideas:
- LocS
- GlobS
- VarS
'''
import numpy as np
from src.magcal import GnosticsCharacteristics

class ScaleParam():
    """
    A Machine Gnostic class to compute and optimize scale parameter for different gnostic distribution functions

    Parameters
    ----------
        F: Input Parameter, e.g., fidelity of the data
    """
    
    def _gscale_loc(self, F):
        '''
        For internal use only

        calculates the local scale parameter for given calculated F at Scale = 1.
        S with be in the same shape as F.
        Solve for scale parameter using Newton-Raphson."
        '''
        m2pi = 2 / np.pi
        sqrt2 = np.sqrt(2)
        
        if F < m2pi * sqrt2 / 3:
            S = np.pi
        elif F < m2pi:
            S = 3 * np.pi / 4
        elif F < m2pi * sqrt2:
            S = np.pi / 2
        else:
            S = np.pi / 4

        epsilon = 1e-5
        for _ in range(100):
            delta = (np.sin(S) - S * F) / (np.cos(S) - F)
            S -= delta
            if abs(delta) < epsilon:
                break
        return S * m2pi