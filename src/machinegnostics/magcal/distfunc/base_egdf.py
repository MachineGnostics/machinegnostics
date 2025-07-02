"""
Machine Gnostics - Machine Gnostics Library
Copyright (C) 2025  Machine Gnostics Team

Author: Nirmal Parmar
License: GNU General Public License v3.0 (GPL-3.0)

# Base Gnostic Distribution Function Module

EGDF - Estimating Global Distribution Function
"""

import numpy as np
import warnings
from machinegnostics.magcal.characteristics import GnosticsCharacteristics
from machinegnostics.magcal.distfunc.base_df import BaseDistFunc
from machinegnostics.magcal.distfunc.base_df_tranformer import BaseDistFuncTransformer
from machinegnostics.magcal.scale_param import ScaleParam

class BaseEGDF(BaseDistFunc, BaseDistFuncTransformer):
    """
    Estimating Global Distribution Function (EGDF) base class.
    """

    def __init__(self,
                 data,
                 DLB: float,
                 DUB: float,
                 LSB: float,
                 USB: float,
                 LB: float,
                 UB: float,
                 S = 1,
                 tolerance: float = 1e-5,
                 data_form: str = 'a',
                 n_points: int = 100,
                 homogeneous: bool = True,
                 catch: bool = True,
                 weights: np.ndarray = None
                 ):
        
        self.data = data
        self.S = S
        self.tolerance = tolerance
        self.DLB = DLB
        self.DUB = DUB
        self.LSB = LSB
        self.USB = USB
        self.LB = LB
        self.UB = UB
        self.data_form = data_form
        self.n_points = n_points
        self.homogeneous = homogeneous
        self.catch = catch
        self.weights = weights
        # to store parameters
        self.param = {}

        # order data
        if not isinstance(self.data, np.ndarray):
            self.data = np.array(self.data)
        # sort data
        self.data = np.sort(self.data)

        # initial parameters
        self.z = None  # transformed data
        self.z_i = None  # inverse transformed data
        # data bounds
        self.DLB, self.DUB = self._estimate_data_bounds()

        # sample weights
        self._estimate_weights()

        # store values
        # saving the initial bounds
        if self.catch:
            self.param['LB'] = self.LB
            self.param['UB'] = self.UB
            self.param['DLB'] = self.DLB
            self.param['DUB'] = self.DUB
            self.param['LSB'] = self.LSB
            self.param['USB'] = self.USB
            self.param['S'] = self.S
            self.param['data'] = self.data
            self.param['data_form'] = self.data_form
            self.param['homogeneous'] = self.homogeneous
        else:
            self.param = None

        # argument validation in main class

        # transform input data in transformer class

    def _fit(self):
        """
        Fit the EGDF to the data.
        """
        # transform input data
        self._transform_input()
        # calculate EGDF
        self._gdf()
        # calculate PDF
        self._pdf()

    def _pdf(self):
        """
        Probability Density Function (PDF) of the EGDF.
        """
        pass

    def _gdf(self):
        """
        Gnostic (cumulative) Distribution Function (GDF) of the EGDF.
        """
        # Calculate CDF values using EGDF formula
        self._compute_cdf()
        if self.catch:
            self.param['egdf'] = self.egdf_values
        else:
            self.param['egdf'] = None
        return self

    def _plot(self):
        """
        Plot the EGDF and PDF.
        """
        # plot EGDF - blue line y1 axis

        # plot PDF - red line y2 axis

        # show bounds and data points - optional argument
        pass

    def _estimate_weights(self):
        """
        Estimate weights for the EGDF.
        
        This method can be overridden in subclasses to provide custom weight estimation logic.
        """
        # Default implementation uses uniform weights
        if self.weights is None:
            self.weights = np.ones_like(self.data)
        else:
            self.weights = np.asarray(self.weights)
            if len(self.weights) != len(self.data):
                raise ValueError("weights must have the same length as data")
        # Normalize weights to sum to n (number of data points)
        self.weights = self.weights / np.sum(self.weights) * len(self.weights)

        # store weights to param
        if self.catch:
            self.param['prior_weights'] = self.weights
        else:
            self.param['prior_weights'] = None

    def _estimate_data_bounds(self):
        """
        Estimate data bounds based on the EGDF.

        DLB and DUB are the data bounds where samples are expected.
        """
        # Estimate data bounds
        if self.DLB is None:
            self.DLB = np.min(self.data)
        if self.DUB is None:
            self.DUB = np.max(self.data)
        return self.DLB, self.DUB

    def _estimate_sample_boundaries(self):
        """
        Estimate sample boundaries based on the EGDF.

        LSB and USB are the inner bounds where samples are expected. (lower outliers)
        """
        pass

    def _estimate_outer_bounds(self):
        """
        Estimate outer bounds for data transformation.

        LB and UB are the outer bounds where improbable samples are expected.
        """
        pass

    def _optimize_global_scale(self):
        """
        Optimize the global scale parameter S.
        """
        # Use GnosticsCharacteristics to calculate fidelities and irrelevances
        gc = GnosticsCharacteristics(R=self.R)
        self.scale = ScaleParam()
        # Handle auto S calculation before getting q values
        if isinstance(self.S, str) and self.S == 'auto':
            # Use temporary S=1.0 to get initial fidelities
            q_temp, q1_temp = gc._get_q_q1(S=1)
            fidelities_temp = gc._fi(q_temp, q1_temp)
            
            # Calculate S based on mean fidelity
            self.S_opt = self.scale._gscale_loc(np.mean(fidelities_temp))

            # is S in nan then return S = 1 with user warning
            if np.isnan(self.S_opt):
                warnings.warn("Calculated S is NaN, using S=1.0 instead", RuntimeWarning)
                self.S_opt = 1.0

            # Recalculate with the optimized S value
            self.q, self.q1 = gc._get_q_q1(S=self.S_opt)
        else:
            # Use the provided S value directly
            self.S = self.S
            self.q, self.q1 = gc._get_q_q1(S=self.S)
        # saving param S
        if self.catch:
            self.param['S'] = self.S
            self.param['S_opt'] = self.S_opt
        else:
            self.param['S'] = None
            self.param['S_opt'] = None
        

    def _homogenize_data(self):
        """
        Homogenize the data if required.
        """
        pass

    def _homogenity_check(self):
        """
        Check if the data is homogeneous.
        """
        pass

    def _calculate_fidelities_irrelevances(self, Z0_w):
        """
        Calculate fidelities and irrelevances for each data point.
        
        Parameters
        ----------
        Z0_w : ndarray
            Transformed evaluation points.
        
        Returns
        -------
        tuple
            fidelities and irrelevances arrays
        """
        # Get gnostic characteristics - use transformed data from the current instance
        Z_working = self.zi.reshape(-1, 1)  # Shape: (n_samples, 1)
        eps = np.finfo(float).eps
        
        # Calculate ratio R = Z/Z0
        # replace R infinity with max float value
        self.R = Z_working / (Z0_w + eps)

        # Use GnosticsCharacteristics to calculate fidelities and irrelevances
        gc = GnosticsCharacteristics(R=self.R)
        self.scale = ScaleParam()

        # optimize global scale parameter S
        self._optimize_global_scale()
        
        # # Handle auto S calculation before getting q values
        # if isinstance(self.S, str) and self.S == 'auto':
        #     # Use temporary S=1.0 to get initial fidelities
        #     q_temp, q1_temp = gc._get_q_q1(S=1)
        #     fidelities_temp = gc._fi(q_temp, q1_temp)
            
        #     # Calculate S based on mean fidelity
        #     self.S_opt = self.scale._gscale_loc(np.mean(fidelities_temp))

        #     # is S in nan then return S = 1 with user warning
        #     if np.isnan(self.S_opt):
        #         warnings.warn("Calculated S is NaN, using S=1.0 instead", RuntimeWarning)
        #         self.S_opt = 1.0

        #     # Recalculate with the optimized S value
        #     self.q, self.q1 = gc._get_q_q1(S=self.S_opt)
        # else:
        #     # Use the provided S value directly
        #     self.S = self.S
        #     self.q, self.q1 = gc._get_q_q1(S=self.S)
        # # saving param S
        # if self.catch:
        #     self.param['S'] = self.S
        #     self.param['S_opt'] = self.S_opt if hasattr(self, 'S_opt') else None
        # else:
        #     self.param['S'] = None
        #     self.param['S_opt'] = None

        fidelities = gc._fi(self.q, self.q1)
        irrelevances = gc._hi(self.q, self.q1)

        return fidelities, irrelevances
    
    def _compute_cdf(self):
        """
        Internal method to compute CDF values using equation 15.29.
        
        Returns
        -------
        ndarray
            Calculated CDF values at each evaluation point using EGDF formula.
        """
        # Generate evaluation points in the infinite domain if not already set
        if not hasattr(self, 'zi_points') or self.zi_points is None:
            # Create evaluation points between the bounds of transformed data
            self.zi_points = np.linspace(np.min(self.zi), np.max(self.zi), self.n_points)
        
        # Reshape evaluation points for broadcasting
        Z0_w = self.zi_points.reshape(1, -1)  # Shape: (1, n_points)
        
        # Reshape weights for broadcasting
        weights = self.weights.reshape(-1, 1)  # Shape: (n_samples, 1)

        # Calculate fidelities and irrelevances
        fidelities, irrelevances = self._calculate_fidelities_irrelevances(Z0_w)
        
        # Calculate weighted means using equation 15.31
        mean_fidelity = np.sum(weights * fidelities, axis=0) / np.sum(weights)  # f̄_E
        mean_irrelevance = np.sum(weights * irrelevances, axis=0) / np.sum(weights)  # h̄_E
        
        # Calculate estimating modulus using equation 15.28
        M_zi = np.sqrt(mean_fidelity**2 + mean_irrelevance**2)
        
        # Calculate EGDF using equation 15.29
        self.egdf_values = (1 - mean_irrelevance / M_zi) / 2

    def _compute_pdf(self):
        """
        Internal method to compute PDF values using equations 15.30 and 15.31.
            
        Returns
        -------
        ndarray
            Calculated PDF values at each evaluation point using EGDF density formula.
        """
        # Use the same evaluation points as CDF
        if not hasattr(self, 'zi_points') or self.zi_points is None:
            # Create evaluation points between the bounds of transformed data
            self.zi_points = np.linspace(np.min(self.zi), np.max(self.zi), self.n_points)
        
        # Initialize output array
        density = np.zeros_like(self.zi_points, dtype=float)

        # Handle empty data case
        if len(self.zi) == 0:
            return density
        
        # Handle zero and near-zero evaluation point values
        eps = np.finfo(float).eps
        mask = np.abs(self.zi_points) > eps
        Z0_safe = self.zi_points[mask]
        
        if len(Z0_safe) > 0:
            # Reshape for broadcasting
            Z0_w = Z0_safe.reshape(1, -1)  # Shape: (1, n_points)
            weights = self.weights.reshape(-1, 1)  # Shape: (n_samples, 1)
            
            # Calculate fidelities and irrelevances
            fidelities, irrelevances = self._calculate_fidelities_irrelevances(Z0_w)
            
            # Calculate weighted means using equation 15.31
            mean_fidelity = np.sum(weights * fidelities, axis=0) / np.sum(weights)  # f̄_E
            mean_irrelevance = np.sum(weights * irrelevances, axis=0) / np.sum(weights)  # h̄_E
            
            # Calculate F2 and FH using equation 15.31
            F2 = np.sum(weights * fidelities**2, axis=0) / np.sum(weights)
            FH = np.sum(weights * fidelities * irrelevances, axis=0) / np.sum(weights)
            
            # Calculate estimating modulus using equation 15.28
            M_zi = np.sqrt(mean_fidelity**2 + mean_irrelevance**2)
            M_zi_cubed = M_zi**3
            
            # Calculate PDF using equation 15.30
            numerator = (mean_fidelity**2) * F2 + mean_fidelity * mean_irrelevance * FH
            density_safe = (1 / (self.S)) * (numerator / M_zi_cubed)
            
            # Assign calculated values to output array
            density[mask] = density_safe
            
            # Handle negative density values (as mentioned in the text, EGDF can have negative density)
            # We could either clip to zero or leave as is based on the application requirements
            if np.any(density < 0):
                warnings.warn("EGDF density contains negative values, which may indicate non-homogeneous data", RuntimeWarning)
        
        self.pdf_values = density
        return density