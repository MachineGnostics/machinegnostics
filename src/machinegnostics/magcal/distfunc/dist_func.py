"""
Machine Gnostics - Machine Gnostics Library
Copyright (C) 2025  Machine Gnostics Team

Author: Nirmal Parmar
License: GNU General Public License v3.0 (GPL-3.0)

# Gnostic Distribution Function Module

This module implements the Gnostic distribution functions, which are a family of robust statistical 
methods based on the principles of information theory.
"""

import numpy as np
from machinegnostics.magcal.data_conversion import DataConversion

class DistributionFunctions:
    """
    Base class for Gnostic distribution functions.
    
    Gnostic distribution functions are a family of robust statistical methods based on 
    the principles of information theory.
    """
    
    def __init__(self, tol=1e-6, data_form='a', data_points=1000):
        """
        Initialize the DistributionFunction.
        
        Parameters
        ----------
        tol : float, optional
            Tolerance for convergence in iterative methods (default is 1e-6).
        data_form : str, optional
            Data form: 'a' for additive, 'm' for multiplicative (default is 'a').
        data_points : int, optional
            Default number of evaluation points to generate (default is 1000).
        """
        self.tol = tol
        self.Z = None    # Original data
        self.Zi = None   # Data in infinite domain
        self.S = None    # Scale parameter
        self.Z0 = None   # Evaluation points in original domain
        self.Z0i = None  # Evaluation points in infinite domain
        self.data_points = data_points
        self.data_form = data_form  # 'a' for additive or 'm' for multiplicative
        
        # Data bounds attributes
        self.data_lb = None
        self.data_ub = None
        self.fin_lb = None
        self.fin_ub = None
        
    def fit(self, Z, S):
        """
        Fit the distribution with data and scale parameter.
        
        Parameters
        ----------
        Z : array-like
            Data points (observations).
        S : float or array-like
            Scale parameter(s) controlling the spread of the distribution.
            
        Returns
        -------
        self : DistributionFunction
            Returns self for method chaining.
        """
        # Store original data
        self.Z = np.asarray(Z)
        
        # Validate input data
        if self.Z.ndim != 1:
            raise ValueError("Z must be a 1D array of data points")
        if len(self.Z) == 0:
            raise ValueError("Z must contain at least one data point")
            
        # Process scale parameter
        if np.isscalar(S):
            self.S = np.full_like(self.Z, S, dtype=float)
        else:
            S = np.asarray(S)
            if len(S) != len(self.Z):
                raise ValueError("Scale array must match data length")
            self.S = S
            
        # Set data bounds from original data
        self.data_lb = self.Z.min()
        self.data_ub = self.Z.max()
            
        # Generate default Z0 points in original domain
        padding = 0.1 * (self.Z.max() - self.Z.min())
        self.Z0 = np.linspace(
            max(self.Z.min() - padding, 1e-10),  # Avoid zero or negative values
            self.Z.max() + padding, 
            self.data_points
        )
        
        # Transform data to working domain if needed
        if self.data_form == 'm':  # Only transform for multiplicative data
            self.Zi = self._to_working_domain(self.Z)
            self.Z0i = self._to_working_domain(self.Z0)
        else:  # For additive data, use original domain
            self.Zi = self.Z
            self.Z0i = self.Z0
            
        return self
    
    def set_evaluation_points(self, Z0=None, data_points=None):
        """
        Set custom evaluation points or number of auto-generated points.
        
        Parameters
        ----------
        Z0 : array-like, optional
            Custom evaluation points in original domain.
        data_points : int, optional
            Number of points to generate if Z0 is None.
            
        Returns
        -------
        self : DistributionFunction
            Returns self for method chaining.
        """
        if Z0 is not None:
            # Use provided evaluation points
            self.Z0 = np.asarray(Z0)
            # Transform to working domain if needed
            if self.data_form == 'm':  # Only transform for multiplicative data
                self.Z0i = self._to_working_domain(self.Z0)
            else:
                self.Z0i = self.Z0
        elif data_points is not None:
            # Update data points count and regenerate points
            self.data_points = data_points
            if self.Z is not None:
                padding = 0.1 * (self.Z.max() - self.Z.min())
                self.Z0 = np.linspace(
                    max(self.Z.min() - padding, 1e-10),
                    self.Z.max() + padding,
                    self.data_points
                )
                # Transform to working domain if needed
                if self.data_form == 'm':  # Only transform for multiplicative data
                    self.Z0i = self._to_working_domain(self.Z0)
                else:
                    self.Z0i = self.Z0
        return self
    
    def _to_working_domain(self, Z):
        """
        Transform data to the working domain based on data form.
        
        For additive data, this is identity.
        For multiplicative data, this transforms to log domain.
        
        Parameters
        ----------
        Z : array-like
            Data in original domain
            
        Returns
        -------
        Z_working : array-like
            Data in working domain
        """
        if Z is None:
            return None
            
        Z = np.asarray(Z)
        
        # For multiplicative data, work in log domain
        if self.data_form == 'm':
            # Ensure positive values
            Z_positive = np.maximum(Z, np.finfo(float).eps)
            return np.log(Z_positive)
        else:
            # For additive data, use original domain
            return Z
    
    def _from_working_domain(self, Z_working):
        """
        Transform data from working domain back to original domain.
        
        Parameters
        ----------
        Z_working : array-like
            Data in working domain
            
        Returns
        -------
        Z : array-like
            Data in original domain
        """
        if Z_working is None:
            return None
            
        Z_working = np.asarray(Z_working)
        
        # For multiplicative data, transform back from log domain
        if self.data_form == 'm':
            return np.exp(Z_working)
        else:
            # For additive data, use original domain
            return Z_working
    
    def gnostic_kernel(self, Z0=None):
        """
        Compute the Gnostic kernel function for each data point.
        
        Parameters
        ----------
        Z0 : array-like, optional
            Points where to evaluate the kernel in original domain. 
            
        Returns
        -------
        kernels : ndarray
            Array of shape (len(Z0), len(Z)) where each column is 
            the kernel for a data point evaluated at all Z0 points.
        """
        if self.Z is None or self.S is None:
            raise ValueError("Must call fit(Z, S) before calculating kernels")
            
        # Use default evaluation points if not provided
        if Z0 is None:
            Z0 = self.Z0
            Z0_working = self.Z0i
        else:
            Z0 = np.asarray(Z0)
            # Transform to working domain if needed
            if self.data_form == 'm':
                Z0_working = self._to_working_domain(Z0)
            else:
                Z0_working = Z0
        
        # Work in appropriate domain based on data form
        Z0_col = Z0_working.reshape(-1, 1)  # Shape: (n_points, 1)
        Z_row = self.Zi.reshape(1, -1)      # Shape: (1, n_samples)
        S_row = self.S.reshape(1, -1)       # Shape: (1, n_samples)
        
        # Compute kernels using broadcasting
        A = Z0_col - Z_row
        cosh_term = np.cosh(2 * A / S_row)
        kernels = 1.0 / (S_row * cosh_term ** 2)
        
        return kernels
    
    def gnostic_density(self, Z0=None, normalize=True):
        """
        Calculate the Gnostic kernel density estimate.
        
        Parameters
        ----------
        Z0 : array-like, optional
            Points where to evaluate the density in original domain.
        normalize : bool, optional
            If True, normalize so the area under density is 1.
            
        Returns
        -------
        density : array-like
            Density values at each point in Z0 in original domain.
        """
        if self.Z is None or self.S is None:
            raise ValueError("Must call fit(Z, S) before calculating density")
            
        # Use default evaluation points if not provided
        if Z0 is None:
            Z0 = self.Z0
        else:
            Z0 = np.asarray(Z0)
        
        # Compute kernels and take mean along axis 1 (samples)
        kernels = self.gnostic_kernel(Z0)
        density = np.mean(kernels, axis=1)
        
        # Normalize if requested
        if normalize and len(density) > 1:
            # For multiplicative data, adjust for log-transformation
            if self.data_form == 'm':
                # Integration in log domain needs adjustment
                area = np.trapz(density * Z0, np.log(Z0))
            else:
                area = np.trapz(density, Z0)
                
            if area > 0:
                density /= area
                
        return density
    
    def eldf(self, Z0=None):
        """
        Compute the Estimating Local Distribution Function (ELDF).
        
        Parameters
        ---------- 
        Z0 : array-like, optional
            The reference points where to evaluate the distribution.
        
        Returns
        -------
        eldf_values : array-like
            The estimated local distribution function values.
        """
        if self.Z is None or self.S is None:
            raise ValueError("Must call fit(Z, S) before calculating ELDF")
            
        # Use default evaluation points if not provided
        if Z0 is None:
            Z0 = self.Z0
            Z0_working = self.Z0i
        else:
            Z0 = np.asarray(Z0)
            # Transform to working domain if needed
            if self.data_form == 'm':
                Z0_working = self._to_working_domain(Z0)
            else:
                Z0_working = Z0
        
        # Work in appropriate domain based on data form
        Z_working = self.Zi.reshape(-1, 1)     # Shape: (n_samples, 1)
        Z0_w = Z0_working.reshape(1, -1)       # Shape: (1, n_points)
        
        # Handle division-by-zero
        eps = np.finfo(float).eps
        Z0_safe = np.maximum(Z0_w, eps)
        
        # Check if all S values are the same
        if np.all(self.S == self.S[0]):
            # Single scalar S - simple broadcasting
            S = self.S[0]
            # Calculate ratio in working domain
            qk = (Z_working / Z0_safe)** (1 / S)
            eldf_values = np.mean(1 / (1 + qk**4), axis=0)
        else:
            # Varying S values - more complex broadcasting
            S = self.S.reshape(-1, 1)  # Shape: (n_samples, 1)
            
            # Calculate ratio in working domain
            qk = (Z_working / Z0_safe)** (1 / S)
            
            # Calculate ELDF values and take mean along samples axis
            eldf_values = np.mean(1 / (1 + qk**4), axis=0)
        
        return eldf_values
    
    def eldf_density(self, Z0=None):
        """
        Calculate the ELDF probability density function.
        
        Parameters
        ----------
        Z0 : array-like, optional
            Grid points at which to evaluate the density.
            
        Returns
        -------
        density : array-like
            Density values for each point in Z0.
        """
        if self.Z is None or self.S is None:
            raise ValueError("Must call fit(Z, S) before calculating density")
            
        # Use default evaluation points if not provided
        if Z0 is None:
            Z0 = self.Z0
            Z0_working = self.Z0i
        else:
            Z0 = np.asarray(Z0)
            # Transform to working domain if needed
            if self.data_form == 'm':
                Z0_working = self._to_working_domain(Z0)
            else:
                Z0_working = Z0
        
        # Handle empty data case
        if len(self.Zi) == 0:
            return np.zeros_like(Z0, dtype=float)
        
        # Initialize output array
        density = np.zeros_like(Z0_working, dtype=float)
        
        # Handle zero and near-zero Z0 values
        eps = np.finfo(float).eps
        mask = np.abs(Z0_working) > eps
        Z0_safe = Z0_working[mask]
        
        if len(Z0_safe) > 0:  # Only proceed if we have valid Z0 values
            # Reshape data for broadcasting
            Z_working = self.Zi.reshape(-1, 1)   # Shape: (n_samples, 1)
            Z0_w = Z0_safe.reshape(1, -1)        # Shape: (1, n_points)
            
            # Check if all S values are the same
            if np.all(self.S == self.S[0]):
                # Vectorized calculation for constant S
                S = self.S[0]
                
                # Calculate q-matrix using the correct formula in working domain
                q_matrix = (Z_working / Z0_w)** (1 / S)
                q_sq = q_matrix**2
                inv_q_sq = 1/q_sq
                
                # Calculate denominator
                denom = (q_sq + inv_q_sq)**2
                
                # Calculate result matrix, handling potential division by zero
                valid_denom = denom > eps
                result = np.zeros_like(denom, dtype=float)
                result[valid_denom] = 4 / denom[valid_denom]
                
                # Average over samples and scale by S
                # The formula is different for multiplicative data
                if self.data_form == 'm':
                    # For multiplicative data in log domain - flatten Z0_w for proper 1D result
                    density_safe = np.mean(result, axis=0) / (S)
                else:
                    # For additive data
                    density_safe = np.mean(result, axis=0) / S
            else:
                # Varying S values need more complex broadcasting
                S = self.S.reshape(-1, 1)  # Shape: (n_samples, 1)
                
                # Calculate q-matrix using the correct formula in working domain
                q_matrix = (Z_working / Z0_w)** (1 / S)
                q_sq = q_matrix**2
                inv_q_sq = 1/q_sq
                
                # Calculate denominator
                denom = (q_sq + inv_q_sq)**2
                
                # Calculate contribution per point
                valid_denom = denom > eps
                result = np.zeros_like(denom, dtype=float)
                
                # Broadcasting S for division
                S_expanded = np.broadcast_to(S, denom.shape)
                
                # Calculate where denominator is valid
                result[valid_denom] = 4 / (S_expanded[valid_denom] * denom[valid_denom])
                
                # Average over samples to get a 1D result
                result_mean = np.mean(result, axis=0)  # This produces a 1D array: (n_points,)
                
                # Now adjust for domain
                if self.data_form == 'm':
                    # For multiplicative data in log domain - make sure to flatten the array
                    # Z0_exp = np.exp(Z0_w).flatten()  # Ensure 1D shape
                    density_safe = result_mean  # This will be 1D
                else:
                    # For additive data
                    density_safe = result_mean  # Already 1D
            
            density[mask] = density_safe
        
        return density
    
    def get_functions(self, Z0=None, normalize=True):
        """
        Get all distribution functions at specified evaluation points.
        
        Parameters
        ----------
        Z0 : array-like, optional
            Points where to evaluate the functions.
        normalize : bool, optional
            If True, normalize densities (default=True).
            
        Returns
        -------
        dict
            Dictionary containing all computed functions.
        """
        if self.Z is None or self.S is None:
            raise ValueError("Must call fit(Z, S) before computing functions")
            
        # Use default evaluation points if not provided
        if Z0 is None:
            Z0 = self.Z0
        else:
            Z0 = np.asarray(Z0)
            # Update Z0i for new Z0
            if self.data_form == 'm':
                Z0i = self._to_working_domain(Z0)
            else:
                Z0i = Z0
        
        # Compute functions
        kernels = self.gnostic_kernel(Z0)
        gnostic_density = self.gnostic_density(Z0, normalize=normalize)
        eldf_values = self.eldf(Z0)
        eldf_density_values = self.eldf_density(Z0)
        
        return {
            'Z0': Z0,                        # Original domain evaluation points 
            'kernels': kernels,              # Kernels at each evaluation point
            'gnostic_density': gnostic_density,   # Gnostic density
            'eldf': eldf_values,             # ELDF values
            'eldf_density': eldf_density_values   # ELDF density
        }