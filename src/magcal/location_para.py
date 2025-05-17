'''
ManGo - Machine Gnostics Library
Copyright (C) 2025  ManGo Team

This work is licensed under the terms of the GNU General Public License version 3.0.

Description: Implementation of Gnostic Median, and intervals calculations
'''

import numpy as np
from scipy.optimize import root_scalar
from src.magcal.characteristics import GnosticsCharacteristics
from src.magcal.scale_param import ScaleParam

class LocationParameter:
    '''
    For internal use only

    Estimates location parameter Z0 (gnostic median), tolerance interval, and interval of typical data
    '''

    def __init__(self,
                 data: np.ndarray,
                 tol=1e-8):
        self.data = data
        self.tol = tol
      
    
    def _gnostic_median(self, case='i', z_range=None):
        """
        Calculate the Gnostic Median of a data sample.
        
        The G-median is defined as the value Z_med for which the sum of irrelevances equals zero.
        Implements both quantifying and estimating cases based on equations 14.23 and 14.24.
                    
        Parameters
        ----------
        data : array-like
            Input data sample
        case : str, default='quantifying'
            The type of G-median to calculate:
            - 'quantifying': Uses equation 14.23
            - 'estimating': Uses equation 14.24
        z_range : tuple, optional
            Initial search range for Z_med (min, max). If None, will be determined from data
        tol : float, default=1e-8
            Tolerance for convergence
            
        Returns
        -------
        float
            The calculated G-median value
            
        References
        ----------
        .. [1] Kovanic P., Humber M.B (2015) The Economics of Information - Mathematical
            Gnostics for Data Analysis. http://www.math-gnostics.eu/books/
        """        
        if z_range is None:
            z_range = (np.min(self.data), np.max(self.data))
        
        def _hc_sum(z_med):
            # define GC
            gc = GnosticsCharacteristics(self.data/z_med)
            q, q1 = gc._get_q_q1()
            
            if case == 'i':
                fi = gc._fi()
                scale = ScaleParam()
                s = scale._gscale_loc(np.mean(fi))
                s = np.where(s > self.tol, s, 1) #NOTE can be improved after
                q, q1 = gc._get_q_q1(S=s)
                hi = gc._hi(q, q1)
                return np.sum(hi)
            elif case == 'j':
                fj = gc._fj()
                scale = ScaleParam()
                s = scale._gscale_loc(np.mean(fj))
                s = np.where(s > self.tol, s, 1) #NOTE can be improved after
                q, q1 = gc._get_q_q1(S=s)
                hj = gc._hi(q, q1)
                return np.sum(hj)
        
        # Find root of irrelevance sum to get G-median
        result = root_scalar(_hc_sum, 
                            bracket=z_range,
                            method='brentq',
                            rtol=self.tol)
        
        if not result.converged:
            raise RuntimeError("G-median calculation did not converge")
            
        return result

