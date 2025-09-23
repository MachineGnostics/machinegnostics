'''
ManGo - Machine Gnostics Library
Copyright (C) 2025  ManGo Team

Author: Nirmal Parmar

This class is deprecated. Use GnosticsCharacteristics class instead.
'''

import numpy as np
from scipy.optimize import root_scalar
from machinegnostics.magcal.characteristics import GnosticsCharacteristics
from machinegnostics.magcal.scale_param import ScaleParam
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
        # If all data is identical, return the common value immediately
        if np.all(self.data == self.data[0]):
            class Result:
                def __init__(self, root):
                    self.root = root
                    self.converged = True
            return Result(self.data[0])
        
        if z_range is None:
            z_range = (np.min(self.data), np.max(self.data))

        z_min, z_max = np.min(self.data), np.max(self.data)
        
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
        # result = root_scalar(_hc_sum, 
        #                     bracket=z_range,
        #                     method='brentq',
        #                     rtol=self.tol)
        
        # Try up to 50% expansion (1% increments) if not converged
        expansion_steps = 5
        expansion_factor = 0.01
        for step in range(expansion_steps + 1):
            try:
                result = root_scalar(_hc_sum, bracket=(z_min, z_max), method='brentq', rtol=self.tol)
                if result.converged:
                    return result
            except Exception:
                pass  # Try expanding the bracket

            # Expand z_min and z_max by 1% each side
            range_width = z_max - z_min
            z_min_exp = z_min - expansion_factor * (step + 1) * range_width
            z_max_exp = z_max + expansion_factor * (step + 1) * range_width
            # Avoid negative or zero z_med if data is strictly positive
            if np.all(self.data > 0):
                z_min = max(z_min_exp, self.tol)
                z_max = max(z_max_exp, z_min + self.tol)
            else:
                z_min = z_min_exp
                z_max = z_max_exp

        raise RuntimeError("G-median calculation did not converge after expanding the bracket by up to 50%.")
    
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
            s = np.where(s > self.tol, s, 1)
            q, q1 = gc._get_q_q1(S=s)
            F = np.mean(gc._fi(q, q1))
            H = np.mean(gc._hi(q, q1))
            c = -1  # For case 'i'
        elif case == 'j':
            # Quantification case
            fj = gc._fj()
            scale = ScaleParam()
            s = scale._gscale_loc(np.mean(fj))
            s = np.where(s > self.tol, s, 1)
            q, q1 = gc._get_q_q1(S=s)
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
        
    def _gnostic_variance(self, data:np.ndarray, case:str = 'i'):
        """
        To calculate gnostic variance of the given sample data.

        For internal use only
        
        """
        data = self.data
        # Validate case parameter
        if case not in ['i', 'j']:
            raise ValueError("case must be either 'i' or 'j'")
        
        z_min, z_max = np.min(data), np.max(data)
        if z_min == z_max:
            return 0
        
        # gmedian
        z0_result = self._gnostic_median(case=case)
        z0 = z0_result.root
        # Get the gnostic characteristics
        gc = GnosticsCharacteristics(data/z0)
        q, q1 = gc._get_q_q1()
        
        # Calculate relevance (F) and irrelevance (H) based on case
        if case == 'i':
            # Estimation case
            fi = gc._fi()
            scale = ScaleParam()
            s = scale._gscale_loc(np.mean(fi))
            s = np.where(s > self.tol, s, 1)
            q, q1 = gc._get_q_q1(S=s)
            H = np.mean(gc._hi(q, q1))
        elif case == 'j':
            # Quantification case
            fj = gc._fj()
            scale = ScaleParam()
            s = scale._gscale_loc(np.mean(fj))
            s = np.where(s > self.tol, s, 1)
            q, q1 = gc._get_q_q1(S=s)
            H = np.mean(gc._hj(q, q1))
        else:
            ValueError("case must be either 'i' or 'j'")
            
        return H
    
    def _gnostic_autocovariance(self, K: int, case: str = 'i') -> float:
        """
        Calculate the gnostic autocovariance according to equation 14.19.
        
        Autocovariance measures the correlation between data points separated by K positions
        within the same data sample.
        
        Parameters
        ----------
        K : int
            Lag parameter, must be between 1 and N-1
        case : str, default='i'
            The type of covariance to calculate:
            - 'i': Estimation case using Hi irrelevance
            - 'j': Quantification case using Hj irrelevance
        
        Returns
        -------
        float
            The calculated autocovariance value
        
        Notes
        -----
        Implementation of equation 14.19:
        acov_c := 1/(N-K) * sum(h_c(2*Omega_i) * h_c(2*Omega_(i+k)))
        where:
        - N is the sample size
        - K is the lag parameter
        - h_c is the irrelevance function (Hi or Hj)
        - Omega_i are the data angles
        
        References
        ----------
        Equation 14.19 in Mathematical Gnostics
        """
        # Validate inputs
        N = len(self.data)
        if not 1 <= K <= N-1:
            raise ValueError(f"K must be between 1 and {N-1}")
        
        # Get G-median for angle calculations
        z0_result = self._gnostic_median(case=case)
        z0 = z0_result.root
        
        # Calculate characteristics
        gc = GnosticsCharacteristics(self.data/z0)
        q, q1 = gc._get_q_q1()
        
        # Get irrelevance values based on case
        if case == 'i':
            h_values = gc._hi(q, q1)
        elif case == 'j':
            h_values = gc._hj(q, q1)
        else:
            raise ValueError("case must be either 'i' or 'j'")
        
        # Calculate autocovariance using equation 14.19
        acov = 0.0
        for i in range(N-K):
            acov += h_values[i] * h_values[i+K]
        
        return acov / (N-K)
    
    def _gnostic_crosscovariance(self, other_data: np.ndarray, case: str = 'i') -> float:
        """
        Calculate the gnostic crosscovariance according to equation 14.20.
    
        Crosscovariance measures the correlation between two different data samples
        of the same size.
    
        Parameters
        ----------
        other_data : np.ndarray
            Second data sample to compare with self.data
        case : str, default='i'
            The type of covariance to calculate:
            - 'i': Estimation case using Hi irrelevance
            - 'j': Quantification case using Hj irrelevance
    
        Returns
        -------
        float
            The calculated crosscovariance value
    
        Notes
        -----
        Implementation of equation 14.20:
        ccov_c := 1/N * sum(h_c(2*Omega_n,A) * h_c(2*Omega_n,B))
        where:
        - N is the sample size
        - h_c is the irrelevance function (Hi or Hj)
        - Omega_n,A and Omega_n,B are angles from samples A and B
    
        References
        ----------
        Equation 14.20 in Mathematical Gnostics
        """
        other_data = np.asarray(other_data)
        if len(self.data) != len(other_data):
            raise ValueError("Both data samples must have the same length")
    
        N = len(self.data)
    
        # Calculate G-medians for both samples
        z0_A = self._gnostic_median(case=case).root
        gcs_B = GnosticCharacteristicsSample(other_data)
        z0_B = gcs_B._gnostic_median(case=case).root
    
        # Calculate characteristics for both samples
        z_A = self.data / z0_A
        z_B = other_data / z0_B
        gc_A = GnosticsCharacteristics(z_A)
        gc_B = GnosticsCharacteristics(z_B)
        q_A, q1_A = gc_A._get_q_q1()
        q_B, q1_B = gc_B._get_q_q1()
    
        # Get irrelevance values based on case (return full arrays, not means)
        if case == 'i':
            fi_A = gc_A._fi(q_A, q1_A)
            scale_A = ScaleParam()
            s_A = scale_A._gscale_loc(np.mean(fi_A))
            s_A = np.where(s_A > self.tol, s_A, 1)
            q_A, q1_A = gc_A._get_q_q1(S=s_A)
            h_values_A = gc_A._hi(q_A, q1_A)
    
            fi_B = gc_B._fi(q_B, q1_B)
            scale_B = ScaleParam()
            s_B = scale_B._gscale_loc(np.mean(fi_B))
            s_B = np.where(s_B > self.tol, s_B, 1)
            q_B, q1_B = gc_B._get_q_q1(S=s_B)
            h_values_B = gc_B._hi(q_B, q1_B)
        elif case == 'j':
            fj_A = gc_A._fj(q_A, q1_A)
            scale_A = ScaleParam()
            s_A = scale_A._gscale_loc(np.mean(fj_A))
            s_A = np.where(s_A > self.tol, s_A, 1)
            q_A, q1_A = gc_A._get_q_q1(S=s_A)
            h_values_A = gc_A._hj(q_A, q1_A)
    
            fj_B = gc_B._fj(q_B, q1_B)
            scale_B = ScaleParam()
            s_B = scale_B._gscale_loc(np.mean(fj_B))
            s_B = np.where(s_B > self.tol, s_B, 1)
            q_B, q1_B = gc_B._get_q_q1(S=s_B)
            h_values_B = gc_B._hj(q_B, q1_B)
        else:
            raise ValueError("case must be either 'i' or 'j'")
    
        # Calculate crosscovariance using equation 14.20
        ccov = np.sum(h_values_A * h_values_B) / N
    
        return ccov
    
    def _gnostic_correlation(self, other_data:np.ndarray, case:str = 'i') -> float:
        '''
        Calculated gnostic correlation from gnostic variance and cross-covariance
        '''
        data = np.asarray(self.data)
        other_data = np.asarray(other_data)

        # If other_data is 2D with one column, reduce to 1D
        if other_data.ndim == 2 and other_data.shape[1] == 1:
            other_data = other_data.ravel()

        # If data is a pandas DataFrame, convert to numpy array
        if hasattr(data, "values"):
            data = data.values

        # If other_data is a pandas DataFrame/Series, convert to numpy array
        if hasattr(other_data, "values"):
            other_data = other_data.values

        # If data is 1D, just compute as before
        if data.ndim == 1:
            d_vars_1 = self._gnostic_variance(case=case, data=data)
            d_vars_2 = self._gnostic_variance(case=case, data=other_data)
            n_ccov_12 = self._gnostic_crosscovariance(other_data=other_data, case=case)
            cor = n_ccov_12 / np.sqrt(d_vars_1 * d_vars_2)
            return cor

        # If data is 2D, compute for each column
        corrs = []
        for i in range(data.shape[1]):
            xi = data[:, i]
            gcs_xi = self.__class__(xi, tol=self.tol)
            d_vars_1 = gcs_xi._gnostic_variance(case=case, data=xi)
            d_vars_2 = gcs_xi._gnostic_variance(case=case, data=other_data)
            n_ccov_12 = gcs_xi._gnostic_crosscovariance(other_data=other_data, case=case)
            cor = n_ccov_12 / np.sqrt(d_vars_1 * d_vars_2)
            corrs.append(cor)
        return np.array(corrs)