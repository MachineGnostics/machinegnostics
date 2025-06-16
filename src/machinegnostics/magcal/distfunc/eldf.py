"""
Machine Gnostics - Machine Gnostics Library
Copyright (C) 2025  Machine Gnostics Team

Author: Nirmal Parmar
License: GNU General Public License v3.0 (GPL-3.0)

# Gnostic Distribution Function Module

ELDF - Estimating Local Distribution Function
"""

import numpy as np
from machinegnostics.magcal.distfunc.data_transform import DataDomainTransformation
from machinegnostics.magcal.distfunc.wedf import WEDF

class ELDF:
    """
    Estimating Local Distribution Function (ELDF)
    
    This class implements the ELDF that estimates the local distribution function
    based on a given set of data points with locality-based weighting.
    """
    
    def __init__(self, data, weights=None, S=1.0, data_form='a', data_lb=None, data_ub=None):
        """
        Initialize the ELDF with data points and optional weights.
        
        Parameters
        ----------
        data : array-like
            Input data values
        weights : array-like, optional
            Weights for each data point. If None, equal weights are used.
        S : float or array-like, optional
            Smoothing parameter(s). Can be a single value or one per data point.
        data_form : str, optional
            Data form: 'a' for additive, 'm' for multiplicative, None for no transformation
        data_lb : float, optional
            Lower bound for the data range. If None, min(data) is used.
        data_ub : float, optional
            Upper bound for the data range. If None, max(data) is used.
        """
        self.data = np.asarray(data)
        
        # Handle weights
        if weights is None:
            self.weights = np.ones_like(self.data)
        else:
            self.weights = np.asarray(weights)
            if len(self.weights) != len(self.data):
                raise ValueError("weights must have the same length as data")
        
        # Normalize weights to sum to n (number of data points)
        self.weights = self.weights / np.sum(self.weights) * len(self.weights)
        
        # Handle smoothing parameter
        if np.isscalar(S):
            self.S = np.full_like(self.data, S)
        else:
            self.S = np.asarray(S)
            if len(self.S) != len(self.data):
                raise ValueError("S must be a scalar or have the same length as data")
        
        # Set data bounds
        if data_lb is None:
            self.data_lb = np.min(self.data)
        else:
            self.data_lb = data_lb
        
        if data_ub is None:
            self.data_ub = np.max(self.data)
        else:
            self.data_ub = data_ub
        
        # Data form and validation
        self.data_form = data_form
        
        if self.data_lb >= self.data_ub:
            raise ValueError("data_lb must be less than data_ub")
        if self.data.size == 0:
            raise ValueError("data must contain at least one element")
        if not np.issubdtype(self.data.dtype, np.number):
            raise ValueError("data must be numeric")
        
        # Initialize data transformer
        self.transformer = DataDomainTransformation(data_form=self.data_form)
        self.transformer.auto_set_bounds(self.data)
        
        # Transform input data to working domain
        self.Z = self.data
        self.Zi = self.transformer.transform_input(self.Z)
        
        # Default evaluation points will be set in fit()
        self.Z0 = None
        self.Z0i = None
        
        # Cache for computed values
        self._cache = {}
    
    def fit(self, Z0=None, n_points=100, compute_pdf=True):
        """
        Compute both ELDF (CDF) and optionally PDF at specified points.
        
        Parameters
        ---------- 
        Z0 : array-like, optional
            The reference points where to evaluate the distribution.
            If None, a linear space between data_lb and data_ub is used.
        n_points : int, optional
            Number of evaluation points if Z0 is None
        compute_pdf : bool, optional
            Whether to compute PDF values in addition to CDF
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'Z0': Evaluation points
            - 'cdf': CDF values at evaluation points
            - 'pdf': PDF values at evaluation points (if compute_pdf=True)
        """
        # Generate evaluation points if not provided
        if Z0 is None:
            self.Z0 = np.linspace(self.data_lb, self.data_ub, n_points)
        else:
            self.Z0 = np.asarray(Z0)
        
        # Transform evaluation points to working domain
        self.Z0i = self.transformer.transform_input(self.Z0)
        
        # Compute CDF values
        cdf_values = self._compute_cdf(self.Z0, self.Z0i)
        
        # Prepare result dictionary
        result = {
            'Z0': self.Z0,
            'cdf': cdf_values
        }
        
        # Compute PDF if requested
        if compute_pdf:
            pdf_values = self._compute_pdf(self.Z0, self.Z0i)
            result['pdf'] = pdf_values
        
        # Cache results
        self._cache[tuple(self.Z0)] = result
        
        return result
    
    def _compute_cdf(self, Z0, Z0i):
        """Internal method to compute CDF values"""
        # Reshape data for broadcasting
        Z_working = self.Zi.reshape(-1, 1)     # Shape: (n_samples, 1)
        Z0_w = Z0i.reshape(1, -1)              # Shape: (1, n_points)
        weights = self.weights.reshape(-1, 1)  # Shape: (n_samples, 1)
        
        # Handle division-by-zero
        eps = np.finfo(float).eps
        Z0_safe = np.maximum(Z0_w, eps)
        
        # Check if all S values are the same
        if np.all(self.S == self.S[0]):
            # Single scalar S - simple broadcasting
            S = self.S[0]
            # Calculate ratio in working domain
            qk = (Z_working / Z0_safe) ** (1 / S)
            # Apply weights to the calculation
            eldf_values = np.sum(weights * (1 / (1 + qk**4)), axis=0) / np.sum(weights)
        else:
            # Varying S values - more complex broadcasting
            S = self.S.reshape(-1, 1)  # Shape: (n_samples, 1)
            
            # Calculate ratio in working domain
            qk = (Z_working / Z0_safe) ** (1 / S)
            
            # Calculate weighted ELDF values
            eldf_values = np.sum(weights * (1 / (1 + qk**4)), axis=0) / np.sum(weights)
        
        return eldf_values
    
    def _compute_pdf(self, Z0, Z0i):
        """Internal method to compute PDF values"""
        # Initialize output array
        density = np.zeros_like(Z0i, dtype=float)
        
        # Handle empty data case
        if len(self.Zi) == 0:
            return density
        
        # Handle zero and near-zero Z0 values
        eps = np.finfo(float).eps
        mask = np.abs(Z0i) > eps
        Z0_safe = Z0i[mask]
        
        if len(Z0_safe) > 0:  # Only proceed if we have valid Z0 values
            # Reshape data for broadcasting
            Z_working = self.Zi.reshape(-1, 1)          # Shape: (n_samples, 1)
            Z0_w = Z0_safe.reshape(1, -1)               # Shape: (1, n_points)
            weights = self.weights.reshape(-1, 1)        # Shape: (n_samples, 1)
            
            # Check if all S values are the same
            if np.all(self.S == self.S[0]):
                # Vectorized calculation for constant S
                S = self.S[0]
                
                # Calculate q-matrix using the correct formula in working domain
                q_matrix = (Z_working / Z0_w) ** (1 / S)
                q_sq = q_matrix**2
                inv_q_sq = 1/q_sq
                
                # Calculate denominator
                denom = (q_sq + inv_q_sq)**2
                
                # Calculate result matrix, handling potential division by zero
                valid_denom = denom > eps
                result = np.zeros_like(denom, dtype=float)
                result[valid_denom] = 4 / denom[valid_denom]
                
                # Apply weights and scale by S
                density_safe = np.sum(weights * result, axis=0) / (np.sum(weights) * S)
                
            else:
                # Varying S values need more complex broadcasting
                S = self.S.reshape(-1, 1)  # Shape: (n_samples, 1)
                
                # Calculate q-matrix using the correct formula in working domain
                q_matrix = (Z_working / Z0_w) ** (1 / S)
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
                
                # Apply weights to get the final density
                density_safe = np.sum(weights * result, axis=0) / np.sum(weights)
            
            density[mask] = density_safe
        
        return density
    
    def cdf(self, Z0=None):
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
        # Check cache first
        if Z0 is not None and tuple(Z0) in self._cache:
            return self._cache[tuple(Z0)]['cdf']
        
        # Compute fit if not in cache
        result = self.fit(Z0, compute_pdf=False)
        return result['cdf']

    def density(self, Z0=None):
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
        # Check if we need to compute PDF (it might not be in cache)
        if Z0 is not None:
            if tuple(Z0) in self._cache:
                if 'pdf' in self._cache[tuple(Z0)]:
                    return self._cache[tuple(Z0)]['pdf']
                else:
                    # PDF wasn't computed for this Z0 - compute now
                    Z0i = self.transformer.transform_input(Z0)
                    pdf_values = self._compute_pdf(Z0, Z0i)
                    self._cache[tuple(Z0)]['pdf'] = pdf_values
                    return pdf_values
        
        # Compute full fit if not cached
        result = self.fit(Z0, compute_pdf=True)
        return result['pdf']
    
    def plot(self, ax=None, cdf=True, pdf=True, Z0=None, n_points=100):
        """
        Plot the ELDF and ELDF density on a single plot with dual y-axes.
    
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
        cdf : bool, optional
            If True, plot the cumulative distribution function (ELDF).
        pdf : bool, optional
            If True, plot the probability density function (ELDF density).
        Z0 : array-like, optional
            Evaluation points. If None, a linear space is used.
        n_points : int, optional
            Number of evaluation points if Z0 is None.
            
        Returns
        -------
        tuple
            (ax, ax2) - The primary and secondary axes
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots()
        
        # Compute both CDF and PDF at once for efficiency
        result = self.fit(Z0, n_points, compute_pdf=pdf)
        Z0 = result['Z0']
        
        lines = []
        labels = []
        
        # Plot CDF on left y-axis
        if cdf:
            cdf_values = result['cdf']
            line1, = ax.plot(Z0, cdf_values, 'b-', label='ELDF (CDF)')
            ax.set_xlabel('Value')
            ax.set_ylabel('CDF', color='blue')
            ax.tick_params(axis='y', labelcolor='blue')
            ax.grid(True, linestyle='--', alpha=0.7)
            lines.append(line1)
            labels.append('ELDF (CDF)')
        
        # Create second y-axis and plot PDF
        if pdf:
            ax2 = ax.twinx()  # Create a second y-axis that shares the same x-axis
            pdf_values = result['pdf']
            line2, = ax2.plot(Z0, pdf_values, 'r-', label='ELDF Density (PDF)')
            ax2.set_ylabel('Density', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            lines.append(line2)
            labels.append('ELDF Density (PDF)')
        else:
            ax2 = None
        
        # Add legend combining both plots
        if lines:
            ax.legend(lines, labels, loc='best')
        
        # Make sure both plots are visible
        fig = ax.figure
        fig.tight_layout()
        
        return ax if not pdf else (ax, ax2)
    
    def mean(self):
        """Calculate the mean of the distribution"""
        Z0 = np.linspace(self.data_lb, self.data_ub, 1000)
        pdf_values = self.density(Z0)
        dx = Z0[1] - Z0[0]
        return np.sum(Z0 * pdf_values) * dx
    
    def variance(self):
        """Calculate the variance of the distribution"""
        Z0 = np.linspace(self.data_lb, self.data_ub, 1000)
        pdf_values = self.density(Z0)
        dx = Z0[1] - Z0[0]
        mean = self.mean()
        return np.sum((Z0 - mean)**2 * pdf_values) * dx
    
    def skewness(self):
        """Calculate the skewness of the distribution"""
        Z0 = np.linspace(self.data_lb, self.data_ub, 1000)
        pdf_values = self.density(Z0)
        dx = Z0[1] - Z0[0]
        mean = self.mean()
        var = self.variance()
        std = np.sqrt(var)
        return np.sum(((Z0 - mean) / std)**3 * pdf_values) * dx
    
    def kurtosis(self):
        """Calculate the excess kurtosis of the distribution"""
        Z0 = np.linspace(self.data_lb, self.data_ub, 1000)
        pdf_values = self.density(Z0)
        dx = Z0[1] - Z0[0]
        mean = self.mean()
        var = self.variance()
        std = np.sqrt(var)
        return np.sum(((Z0 - mean) / std)**4 * pdf_values) * dx - 3
    
