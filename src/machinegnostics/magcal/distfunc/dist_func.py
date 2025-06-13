"""
Machine Gnostics - Machine Gnostics Library
Copyright (C) 2025  Machine Gnostics Team

Author: Nirmal Parmar
License: GNU General Public License v3.0 (GPL-3.0)

# Gnostic Distribution Function Module

This module implements the Gnostic distribution functions, which are a family of robust statistical 
methods based on the principles of information theory.

Mathematical Gnostics provides a framework for analyzing and interpreting data through the lens of
information theory, allowing for the estimation of distribution functions and densities in 
a non-parametric way.

It includes implementations for:
- Gnostic kernel functions
- Kernel density estimation
- Distribution Function (**DF)
- **DF density computation

The Gnostic distribution functions are designed to be flexible and robust, allowing for the analysis of
data in both additive and multiplicative domains. The methods are implemented using vectorized operations

- ELDF: Estimating Local Distribution Function
- EGDF: Estimating Global Distribution Function
- QLDF: Quantification Local Distribution Function
- QGDF: Quantification Global Distribution Function
- Gnostic Kernel: A kernel function used in Gnostic methods
- Gnostic Density: A non-parametric density estimate using Gnostic kernels
"""

import numpy as np
from machinegnostics.magcal.scale_param import ScaleParam
from machinegnostics.magcal.characteristics import GnosticsCharacteristics
from machinegnostics.magcal.data_conversion import DataConversion

class DistributionFunction:
    """
    Base class for Gnostic distribution functions.
    
    Gnostic distribution functions are a family of robust statistical methods based on 
    the principles of information theory. This class provides implementations of various
    Gnostic distribution functions and related statistics, including:
    
    - Gnostic kernel functions
    - Kernel density estimation
    - Estimating Local Distribution Function (ELDF)
    - ELDF density computation
    
    The class uses a consistent naming convention for attributes:
    - Z: Input data points in original domain
    - Zi: Input data points transformed to infinite domain
    - S: Scale parameter(s) controlling the spread of the distribution
    - Z0: Reference points in original domain
    - Z0i: Reference points transformed to infinite domain
    
    References
    ----------
    Kovanic, P., & Humber, M. B. (2015). The Economics of Information: 
    Mathematical Gnostics for Data Analysis. 
    """
    
    def __init__(self, tol=1e-6, data_form:str='a', data_points:int=1000):
        """
        Initialize the GnosticDistributionFunction with convergence tolerance.
        
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
        self.data_lb = None  # Lower bound of original data
        self.data_ub = None  # Upper bound of original data
        self.fin_lb = None   # Lower bound of normalized data
        self.fin_ub = None   # Upper bound of normalized data
        
    def fit(self, Z, S):
        """
        Fit the distribution with data and scale parameter.
        
        Parameters
        ----------
        Z : array-like
            Data points (observations).
        S : float or array-like
            Scale parameter(s) controlling the spread of the distribution.
            Can be a scalar (same for all points) or array of same length as Z.
            
        Returns
        -------
        self : GnosticDistributionFunction
            Returns self for method chaining.
        """
        # Store original data
        self.Z = np.asarray(Z)
        
        # Validate input data
        if self.Z.ndim != 1:
            raise ValueError("Z must be a 1D array of data points")
        if len(self.Z) == 0:
            raise ValueError("Z must contain at least four data points")
        if len(self.Z) < 4:
            raise ValueError("Z must contain at least four data points")
            
        # Process scale parameter
        if np.isscalar(S):
            self.S = np.full_like(self.Z, S, dtype=float)
        else:
            S = np.asarray(S)
            if len(S) != len(self.Z):
                raise ValueError("Scale array must match data length")
            self.S = S
            
        # Generate default Z0 points in original domain
        padding = 0.1 * (self.Z.max() - self.Z.min())
        self.Z0 = np.linspace(
            max(self.Z.min() - padding, 1e-10),  # Avoid zero or negative values
            self.Z.max() + padding, 
            self.data_points
        )
        
        # Transform data to infinite domain
        # self.Zi = self._transform_to_infinite(self.Z)
        self.Zi = Z
        
        # Transform evaluation points to infinite domain
        # self.Z0i = self._transform_to_infinite(self.Z0)
        self.Z0i = self.Z0
        
        return self
    
    def set_evaluation_points(self, Z0=None, data_points=None):
        """
        Set custom evaluation points or number of auto-generated points.
        
        Parameters
        ----------
        Z0 : array-like, optional
            Custom evaluation points in original domain. If None, points will be auto-generated.
        data_points : int, optional
            Number of points to generate if Z0 is None. If None, uses default.
            
        Returns
        -------
        self : GnosticDistributionFunction
            Returns self for method chaining.
        """
        if Z0 is not None:
            # Use provided evaluation points
            self.Z0 = np.asarray(Z0)
            # Transform to infinite domain
            self.Z0i = self._transform_to_infinite(self.Z0)
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
                # Transform to infinite domain
                self.Z0i = self._transform_to_infinite(self.Z0)
        return self
    
    def _transform_to_infinite(self, Z):
        """
        Transform data from original domain to infinite domain.
        
        Parameters
        ----------
        Z : array-like
            Data in original domain
            
        Returns
        -------
        Zi : array-like
            Data transformed to infinite domain
        """
        if Z is None:
            return None
            
        dc = DataConversion()
        Z = np.asarray(Z)
        
        # On first call, store original data bounds
        if self.data_lb is None or self.data_ub is None:
            self.data_lb = self.Z.min()
            self.data_ub = self.Z.max()
        
        # Convert to standard normalized form based on data type
        if self.data_form == 'a':  # Additive data
            zZ = dc._convert_az(Z, lb=self.data_lb, ub=self.data_ub)
        elif self.data_form == 'm':  # Multiplicative data
            zZ = dc._convert_mz(Z, lb=self.data_lb, ub=self.data_ub)
        else:
            raise ValueError("data_form must be 'a' (additive) or 'm' (multiplicative)")
        
        # On first call, store finite domain bounds
        if self.fin_lb is None or self.fin_ub is None:
            # Use bounds from data points, not evaluation points
            data_zZ = dc._convert_az(self.Z, lb=self.data_lb, ub=self.data_ub) if self.data_form == 'a' else \
                      dc._convert_mz(self.Z, lb=self.data_lb, ub=self.data_ub)
            self.fin_lb = data_zZ.min()
            self.fin_ub = data_zZ.max()
        
        # Transform from finite to infinite domain
        Zi = dc._convert_fininf(zZ, lb=self.fin_lb, ub=self.fin_ub)
        
        return Zi
    
    def _transform_from_infinite(self, Zi):
        """
        Transform data from infinite domain back to original domain.
        
        Parameters
        ----------
        Zi : array-like
            Data in infinite domain
            
        Returns
        -------
        Z : array-like
            Data transformed back to original domain
        """
        if Zi is None:
            return None
            
        dc = DataConversion()
        Zi = np.asarray(Zi)
        
        # Check if bounds are available
        if self.fin_lb is None or self.fin_ub is None or self.data_lb is None or self.data_ub is None:
            raise ValueError("Cannot transform data back without prior fit")
        
        # Transform from infinite back to finite domain
        Z_fin = dc._convert_inffin(Zi, lb=self.fin_lb, ub=self.fin_ub)
        
        # Transform from standard normalized form to original domain
        if self.data_form == 'a':  # Additive data
            Z_orig = dc._convert_za(Z_fin, lb=self.data_lb, ub=self.data_ub)
        elif self.data_form == 'm':  # Multiplicative data
            Z_orig = dc._convert_zm(Z_fin, lb=self.data_lb, ub=self.data_ub)
        else:
            raise ValueError("data_form must be 'a' (additive) or 'm' (multiplicative)")
        
        return Z_orig
    
    def gnostic_kernel(self, Z0=None):
        """
        Compute the Gnostic kernel function for each data point using vectorized operations.
        
        The Gnostic kernel is defined as 1/(S * cosh(2(Z-Z0)/S)²) for additive data or
        1/(S * cosh(2*log(Z0/Z)/S)²) for multiplicative data.
        
        Parameters
        ----------
        Z0 : array-like, optional
            Points where to evaluate the kernel in original domain. 
            If None, uses self.Z0.
            
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
            Z0i = self.Z0i
        else:
            Z0 = np.asarray(Z0)
            Z0i = self._transform_to_infinite(Z0)
        
        # Work in infinite domain for calculations
        Z0i_col = Z0i.reshape(-1, 1)  # Shape: (n_points, 1)
        Zi_row = self.Zi.reshape(1, -1)  # Shape: (1, n_samples)
        S_row = self.S.reshape(1, -1)  # Shape: (1, n_samples)
        
        # Compute kernels using broadcasting (creates a matrix of shape (n_points, n_samples))
        A = Z0i_col - Zi_row  # Difference in infinite domain
        cosh_term = np.cosh(2 * A / S_row)  # Element-wise division and cosh
        kernels = 1.0 / (S_row * cosh_term ** 2)  # Element-wise multiplication
        
        return kernels
    
    def gnostic_density(self, Z0=None, normalize=True):
        """
        Calculate the Gnostic kernel density estimate using vectorized operations.
        
        This method computes a non-parametric density estimate by placing a Gnostic
        kernel at each data point and summing their contributions.
        
        Parameters
        ----------
        Z0 : array-like, optional
            Points where to evaluate the density in original domain.
            If None, uses self.Z0.
        normalize : bool, optional
            If True, normalize so the area under density is 1 (default=True).
            
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
            area = np.trapz(density, Z0)
            if area > 0:
                density /= area
                
        return density
    
    def eldf(self, Z0=None):
        """
        Compute the Estimating Local Distribution Function (ELDF) using matrix operations.
        
        ELDF is a robust distribution function that maps input data to 
        a distribution estimate at reference points.
        
        Parameters
        ---------- 
        Z0 : array-like, optional
            The reference points where to evaluate the distribution in original domain.
            If None, uses self.Z0.
        
        Returns
        -------
        eldf_values : array-like
            The estimated local distribution function values at points Z0 in original domain.
        """
        if self.Z is None or self.S is None:
            raise ValueError("Must call fit(Z, S) before calculating ELDF")
            
        # Use default evaluation points if not provided
        if Z0 is None:
            Z0 = self.Z0
            Z0i = self.Z0i
        else:
            Z0 = np.asarray(Z0)
            Z0i = self._transform_to_infinite(Z0)
        
        # Work in infinite domain for calculations
        Zi = self.Zi.reshape(-1, 1)  # Shape: (n_samples, 1)
        Z0i = Z0i.reshape(1, -1)     # Shape: (1, n_points)
        
        # Handle division-by-zero for Z0 near zero
        eps = np.finfo(float).eps
        Z0i_safe = np.maximum(Z0i, eps)
        
        # Check if all S values are the same
        if np.all(self.S == self.S[0]):
            # Single scalar S - simple broadcasting
            S = self.S[0]
            qk = (Zi / Z0i_safe)**(1/S)
            eldf_infinite = np.mean(1 / (1 + qk**4), axis=0)
        else:
            # Varying S values - more complex broadcasting
            S = self.S.reshape(-1, 1)  # Shape: (n_samples, 1)
            
            # Calculate q-values using broadcasting
            # This creates a matrix of shape (n_samples, n_points)
            qk = np.power(Zi / Z0i_safe, 1/S)
            
            # Calculate ELDF values and take mean along samples axis
            eldf_infinite = np.mean(1 / (1 + qk**4), axis=0)
        
        # Transform result back to original domain
        return eldf_infinite
    
    def eldf_density(self, Z0=None):
        """
        Calculate the ELDF probability density function (dELDF/dZ0) using matrix operations.
        
        This method computes the derivative of the ELDF with respect to Z0,
        providing a probability density function based on the Gnostic approach.
        
        Parameters
        ----------
        Z0 : array-like, optional
            Grid points at which to evaluate the density in original domain.
            If None, uses self.Z0.
            
        Returns
        -------
        density : array-like
            Density values for each point in Z0 in original domain.
        """
        if self.Z is None or self.S is None:
            raise ValueError("Must call fit(Z, S) before calculating density")
            
        # Use default evaluation points if not provided
        if Z0 is None:
            Z0 = self.Z0
            Z0i = self.Z0i
        else:
            Z0 = np.asarray(Z0)
            Z0i = self._transform_to_infinite(Z0)
        
        # Handle empty data case
        if len(self.Zi) == 0:
            return np.zeros_like(Z0, dtype=float)
        
        # Initialize output array
        density = np.zeros_like(Z0i, dtype=float)
        
        # Handle zero and near-zero Z0 values
        eps = np.finfo(float).eps
        mask = np.abs(Z0i) > eps
        Z0i_safe = Z0i[mask]
        
        if len(Z0i_safe) > 0:  # Only proceed if we have valid Z0 values
            # Reshape data for broadcasting
            Zi = self.Zi.reshape(-1, 1)  # Shape: (n_samples, 1)
            Z0i_m = Z0i_safe.reshape(1, -1)  # Shape: (1, n_points)
            
            # Check if all S values are the same
            if np.all(self.S == self.S[0]):
                # Vectorized calculation for constant S
                S = self.S[0]
                
                # Calculate q-matrix and squared q-matrix
                q_matrix = np.abs(Zi / Z0i_m)**(1/S)
                q_sq = q_matrix**2
                inv_q_sq = 1/q_sq
                
                # Calculate denominator
                denom = (q_sq + inv_q_sq)**2
                
                # Calculate result matrix, handling potential division by zero
                valid_denom = denom > eps
                result = np.zeros_like(denom, dtype=float)
                result[valid_denom] = 4 / denom[valid_denom]
                
                # Average over samples and scale by S
                density_safe = np.mean(result, axis=0) / S
            else:
                # Varying S values need more complex broadcasting
                S = self.S.reshape(-1, 1)  # Shape: (n_samples, 1)
                
                # Calculate q-matrix and squared q-matrix - shape (n_samples, n_points)
                q_matrix = np.power(np.abs(Zi / Z0i_m), 1/S)
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
                
                # Average over samples
                density_safe = np.mean(result, axis=0)
            
            density[mask] = density_safe
        
        # Transform result back to original domain
        # density_orig = self._transform_from_infinite(density)
        density_orig = density

        return density_orig
    
    def get_functions(self, Z0=None, normalize=True):
        """
        Get all distribution functions at specified evaluation points.
        
        Parameters
        ----------
        Z0 : array-like, optional
            Points where to evaluate the functions in original domain.
            If None, uses self.Z0.
        normalize : bool, optional
            If True, normalize densities (default=True).
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'Z0': Evaluation points in original domain
            - 'Z0i': Evaluation points in infinite domain
            - 'kernels': Gnostic kernels
            - 'gnostic_density': Gnostic density
            - 'eldf': ELDF values
            - 'eldf_density': ELDF density values
        """
        if self.Z is None or self.S is None:
            raise ValueError("Must call fit(Z, S) before computing functions")
            
        # Use default evaluation points if not provided
        if Z0 is None:
            Z0 = self.Z0
            Z0i = self.Z0i
        else:
            Z0 = np.asarray(Z0)
            Z0i = self._transform_to_infinite(Z0)
        
        # Compute functions
        kernels = self.gnostic_kernel(Z0)
        gnostic_density = self.gnostic_density(Z0, normalize=normalize)
        eldf_values = self.eldf(Z0)
        eldf_density_values = self.eldf_density(Z0)
        
        return {
            'Z0': Z0,                       # Original domain evaluation points
            'Z0i': Z0i,                     # Infinite domain evaluation points
            'kernels': kernels,             # Kernels at each evaluation point
            'gnostic_density': gnostic_density,  # Gnostic density
            'eldf': eldf_values,            # ELDF values
            'eldf_density': eldf_density_values  # ELDF density
        }