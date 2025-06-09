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
        
    import numpy as np

    def var_s(self, Z, W=None, S=1):
        """
        Calculates vector of scale parameters for each kernel.
        
        Parameters:
        Z (array-like): Data vector
        W (array-like, optional): Weight vector
        S (float, optional): Scalar scale factor (default is 1)
        
        Returns:
        numpy.ndarray: Scale vector (same length as Z)
        """
        Z = np.asarray(Z).reshape(-1, 1)

        if W is None:
            W = np.ones_like(Z) / len(Z)
        else:
            W = np.asarray(W).reshape(-1, 1)
            if len(Z) != len(W):
                raise ValueError("Z and W must be of the same length")
            W = W / np.sum(W)

        Sz = np.zeros_like(Z, dtype=float)

        for k in range(len(W)):
            V = Z / Z[k]
            V = V ** 2 + 1.0 / (V ** 2)
            Sz[k] = self._gscale_loc(np.sum(2.0 / V * W))

        Sx = S * Sz / np.mean(Sz)
        return Sx
