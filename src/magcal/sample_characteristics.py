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
import warnings

class GnosticCharacteristicsSample:
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
    
    def _calculate_modulus(self, case='i'):
        """
        Calculate the modulus of the data sample using equation 14.8: M_Z,c = sqrt(F_c^2 - c^2*H_c^2)
        
        Parameters
        ----------
        case : str, default='i'
            The type of modulus to calculate:
            - 'i': Uses irrelevance Hi (estimation case)
            - 'j': Uses irrelevance Hj (quantification case)
            
        Returns
        -------
        float
            The calculated modulus value M_Z,c
            
        Notes
        -----
        This implementation follows Theorem 15 from the reference, which states that
        the modulus of a data sample can be calculated using the relation:
        M_Z,c = sqrt(F_c^2 - c^2*H_c^2)
        
        where:
        - F_c is the relevance function
        - H_c is the irrelevance function
        - c is the case parameter ('i' or 'j')
        
        References
        ----------
        Equation 14.8 in Mathematical Gnostics
        """
        # Validate case parameter
        if case not in ['i', 'j']:
            raise ValueError("case must be either 'i' or 'j'")
        
        z_min, z_max = np.min(self.data), np.max(self.data)
        if z_min == z_max:
            return 1
        
        # gmedian
        z0_result = self._gnostic_median(case=case)
        z0 = z0_result.root
        # Get the gnostic characteristics
        gc = GnosticsCharacteristics(self.data/z0)
        q, q1 = gc._get_q_q1()
        
        # Calculate relevance (F) and irrelevance (H) based on case
        if case == 'i':
            # Estimation case
            fi = gc._fi()
            scale = ScaleParam()
            s = scale._gscale_loc(np.mean(fi))
            # s = np.where(s > self.tol, s, 1)
            # q, q1 = gc._get_q_q1(S=s)
            F = np.mean(gc._fi(q, q1))
            H = np.mean(gc._hi(q, q1))
            c = -1  # For case 'i'
        elif case == 'j':
            # Quantification case
            fj = gc._fj()
            scale = ScaleParam()
            s = scale._gscale_loc(np.mean(fj))
            # s = np.where(s > self.tol, s, 1)
            # q, q1 = gc._get_q_q1(S=s)
            F = np.mean(gc._fj(q, q1))
            H = np.mean(gc._hj(q, q1))
            c = 1  # For case 'j'
        else:
            ValueError("case must be either 'i' or 'j'")
            
        # Calculate modulus using equation 14.8
        M_Z = np.sqrt(np.abs(F**2 - (c**2 * H**2)))
        return M_Z
       
    def _calculate_detailed_modulus(self, Z0, S=None, case='i'): # NOTE not in current use
        """
        Calculate the detailed modulus of the data sample using equation 14.12:
        M_Z,c = sqrt(1 + (c^2/N^2) * sum((f_k*f_l)^(1-c)/2 * ((Z_k/Z_l)^(1/S) - (Z_l/Z_k)^(1/S)))
        
        Parameters
        ----------
        Z0 : float
            Location parameter (usually the G-median)
        S : float, optional
            Scale parameter. If None, will be calculated from data
        case : str, default='i'
            The type of modulus to calculate:
            - 'i': Uses irrelevance Hi (estimation case)
            - 'j': Uses irrelevance Hj (quantification case)
                
        Returns
        -------
        float
            The calculated detailed modulus value M_Z,c
        
        Notes
        -----
        This implementation follows equation 14.12 which provides a more detailed
        calculation of the modulus when all data in the sample Z have Z_0,k = Z_0
        and S_k = S conditions.
        """
        # Input validation
        if case not in ['i', 'j']:
            raise ValueError("case must be either 'i' or 'j'")
        
        # Get gnostic characteristics
        gc = GnosticsCharacteristics(self.data)
        
        # Get scale parameter if not provided
        if S is None:
            if case == 'i':
                fi = gc._fi()
                scale = ScaleParam()
                S = scale._gscale_loc(np.mean(fi))
            else:
                fj = gc._fj()
                scale = ScaleParam()
                S = scale._gscale_loc(np.mean(fj))
        
        # Ensure S is positive and above tolerance
        S = max(S, self.tol)
        
        # Get number of samples
        N = len(self.data)
        
        # Set c based on case
        c = -1 if case == 'i' else 1
        
        # Calculate f_k values based on case
        if case == 'i':
            f_values = gc._fi()
        else:
            f_values = gc._fj()
        
        # Initialize sum
        sum_term = 0.0
        
        # Calculate double sum term
        for k in range(N):
            for l in range(N):
                # Calculate f_k * f_l term
                f_product = f_values[k] * f_values[l]
                
                # Calculate power term (f_k*f_l)^((1-c)/2)
                f_power = np.power(f_product, (1-c)/2)
                
                # Calculate Z_k/Z_l and Z_l/Z_k terms
                Z_ratio_k_l = self.data[k] / self.data[l]
                Z_ratio_l_k = 1 / Z_ratio_k_l
                
                # Calculate the difference term
                diff_term = (np.power(Z_ratio_k_l, 1/S) - 
                            np.power(Z_ratio_l_k, 1/S))
                
                # Add to sum
                sum_term += f_power * diff_term
        
        # Calculate final modulus using equation 14.12
        try:
            M_Z = np.sqrt(1 + (c**2 / N**2) * sum_term)
            
            # Handle potential numerical issues
            if np.isnan(M_Z) or np.isinf(M_Z):
                warnings.warn("Invalid modulus value encountered. Returning 0.0")
                return 0.0
                
            return float(M_Z)
        except ValueError as e:
            warnings.warn(f"Error in modulus calculation: {str(e)}. Returning 0.0")
            return 0.0