"""
Machine Gnostics - Machine Gnostics Library
Copyright (C) 2025  Machine Gnostics Team

Author: Nirmal Parmar
License: GNU General Public License v3.0 (GPL-3.0)

# Gnostic Distribution Function Module

EGDF - Estimating Global Distribution Function
"""

import numpy as np
import warnings
from machinegnostics.magcal.distfunc.data_transform import DataDomainTransformation
from machinegnostics.magcal.scale_param import ScaleParam
from machinegnostics.magcal.mg_weights import GnosticsWeights
from machinegnostics.magcal.characteristics import GnosticsCharacteristics

class EGDF:
    """
    Estimating Global Distribution Function (EGDF)
    
    A gnostic approach to probability distribution estimation that represents a homogeneous
    data sample as a single gnostic event. EGDF provides both cumulative distribution function
    (CDF) and probability density function (PDF) estimations based on input data.
    
    This distribution function is suitable only for data samples that are homogeneous and have
    non-negative density over their full range with only one maximum. EGDF can be used to test
    the homogeneity of a data sample, revealing important features of the data.
    
    EGDF suppresses the influence of "peripheral" data (those with large uncertainties) and 
    focuses on the "central" or "inner" data, for which the fidelities are close to 1 and 
    the irrelevances tend toward zero.
    
    Attributes
    ----------
    data : ndarray
        Input data values
    weights : ndarray
        Normalized weights for each data point
    data_lb : float
        Lower bound for the data range
    data_ub : float
        Upper bound for the data range
    data_form : str
        Data form type ('a' for additive, 'm' for multiplicative, None for no transformation)
    transformer : DataDomainTransformation
        Transformer object for data domain conversions
    S : float or str
        Scale parameter that controls the estimation process, or 'auto' for automatic calculation
    sparam : float
        The computed scale parameter after fitting
    homogeneous : bool
        Whether to use provided weights directly or calculate homogeneous gnostic weights
    """
    
    def __init__(self, 
                data, 
                weights=None, 
                S=1.0, 
                data_form='a', 
                data_lb=None, 
                data_ub=None,
                lb=None,
                ub=None,
                homogeneous:bool=True):
        """
        Initialize the EGDF with data points and optional weights.
        
        Parameters
        ----------
        data : array-like
            Input data values for distribution estimation.
        weights : array-like, optional
            Weights for each data point. If None, equal weights are used.
        S : float or str, optional
            Smoothing parameter controlling the estimation. Default is 1.0.
            Can also be 'auto' for automatic calculation based on data fidelities.
            Recommended range is (0, 2] for numerical stability.
        data_form : str, optional
            Data form specification:
            - 'a' for additive (default)
            - 'm' for multiplicative
            - None for no transformation
        data_lb : float, optional
            Lower bound for the data range. If None, min(data) is used.
        data_ub : float, optional
            Upper bound for the data range. If None, max(data) is used.
        homogeneous : bool, optional
            If True (default), uses the provided weights directly.
            If False, calculates homogeneous gnostic weights based on the data,
            which can provide more robust estimation for non-homogeneous distributions.
        """
        self.data = np.asarray(data)
        
        # homogeneous check
        self.homogeneous = homogeneous
        if not isinstance(self.homogeneous, bool):
            raise ValueError("homogeneous must be a boolean value")
        
        if weights is None:
            self.weights = np.ones_like(self.data)
        else:
            self.weights = np.asarray(weights)
            if len(self.weights) != len(self.data):
                raise ValueError("weights must have the same length as data")

        # Normalize weights to sum to n (number of data points)
        self.weights = self.weights / np.sum(self.weights) * len(self.weights) 
       
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
        # data should be 1-D array
        if self.data.ndim > 1:
            raise ValueError('data should be 1-D array')
        
        # Initialize data transformer
        self.transformer = DataDomainTransformation(data_form=self.data_form, lb=self.data_lb, ub=self.data_ub)
        self.transformer.auto_set_bounds(self.data)
        
        # Transform input data to working domain
        if self.data_form == 'a':
            self.Z = self.transformer._convert_az(self.data, self.data_lb, self.data_ub)
        elif self.data_form == 'm':
            self.Z = self.transformer._convert_mz(self.data, self.data_lb, self.data_ub)
        elif self.data_form is None:
            self.Z = self.data
        else:
            raise ValueError("Invalid data form specified. Use 'a', 'm', or None.")
        # inf bounds
        if lb is None:
            self.lb = np.min(self.Z)
        else:
            self.lb = lb
        if ub is None:
            self.ub = np.max(self.Z)
        else:
            self.ub = ub
        self.Zi = self.transformer._convert_fininf(self.Z, self.lb, self.ub)
        
        # Default evaluation points will be set in fit()
        self.Z0 = None
        self.Z0i = None

        # if homogeneous is False, use gnostic weights
        if not self.homogeneous:
            gw = GnosticsWeights()
            homogeneous_weights = gw._get_gnostic_weights(self.Z)
            self.weights = homogeneous_weights

        # S parameter initialization and validation
        self.scale = ScaleParam()
        self.S = S
        
        # Validate S parameter
        if isinstance(self.S, str):
            self.S = self.S.lower()
            if self.S != 'auto':
                raise ValueError("If S is a string, it must be 'auto'")
        elif isinstance(self.S, (int, float)):
            self.S = float(self.S)
            if self.S <= 0 or self.S > 2:
                warnings.warn("S must be in the range (0, 2] for stability", RuntimeWarning)
        else:
            raise ValueError("S must be a numeric value or 'auto'")
            
        # Initialize the calculated scale parameter
        self.sparam = self.S if not isinstance(self.S, str) else None
        
        # Cache for computed values
        self._cache = {}
    
    def fit(self, Z0=None, n_points=100, compute_pdf=True):
        """
        Compute both EGDF (CDF) and optionally PDF at specified points.
        
        Parameters
        ---------- 
        Z0 : array-like, optional
            The reference points where to evaluate the distribution.
            If None, a linear space between data_lb and data_ub with n_points
            is used.
        n_points : int, optional
            Number of evaluation points if Z0 is None. Default is 100.
        compute_pdf : bool, optional
            Whether to compute PDF values in addition to CDF. Default is True.
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'Z0': ndarray, evaluation points
            - 'cdf': ndarray, CDF values at evaluation points
            - 'pdf': ndarray, PDF values at evaluation points (if compute_pdf=True)
            - 'S': float, the scale parameter used in the calculation
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
        # transform CDF values to original domain
        # cdf_values = self.transformer.transform_output(cdf_values)
        # replace nan with zero
        cdf_values = np.nan_to_num(cdf_values, nan=0.0)
        # replace infinity with 0
        cdf_values = np.where(np.isinf(cdf_values), 0.0, cdf_values)

        # Prepare result dictionary
        result = {
            'Z0': self.Z0,
            'cdf': cdf_values,
            'S': self.sparam,
            'Z0i': self.Z0i
        }
        
        # Compute PDF if requested
        if compute_pdf:
            pdf_values = self._compute_pdf(self.Z0, self.Z0i)
            # replace nan or infinity with zero
            pdf_values = np.nan_to_num(pdf_values, nan=0.0)
            pdf_values = np.where(np.isinf(pdf_values), 0.0, pdf_values)
            # transform PDF values to original domain
            # pdf_values = self.transformer.transform_output(pdf_values)
            result['pdf'] = pdf_values
        
        # Cache results
        self._cache[tuple(self.Z0)] = result
        
        return result
    
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
        # Get gnostic characteristics
        Z_working = self.Zi.reshape(-1, 1)  # Shape: (n_samples, 1)
        eps = np.finfo(float).eps
        
        # Calculate ratio R = Z/Z0
        # replace R infinity with max float value
        R = Z_working / (Z0_w + eps)
        # R = np.where(np.isinf(R), np.finfo(float).max, R)  # Handle division by zero
        
        
        # Use GnosticsCharacteristics to calculate fidelities and irrelevances
        gc = GnosticsCharacteristics(R=R)
        
        # Handle auto S calculation before getting q values
        if isinstance(self.S, str) and self.S == 'auto':
            # Use temporary S=1.0 to get initial fidelities
            q_temp, q1_temp = gc._get_q_q1(S=1.0)
            fidelities_temp = gc._fi(q_temp, q1_temp)
            
            # Calculate S based on mean fidelity
            self.sparam = self.scale._gscale_loc(np.mean(fidelities_temp))

            # is S in nan then return S = 1 with user warning
            if np.isnan(self.sparam):
                warnings.warn("Calculated S is NaN, using S=1.0 instead", RuntimeWarning)
                self.sparam = 1.0
            
            # Recalculate with the proper S value
            q, q1 = gc._get_q_q1(S=self.sparam)
        else:
            # Use the provided S value directly
            self.sparam = self.S
            q, q1 = gc._get_q_q1(S=self.sparam)
            
        fidelities = gc._fi(q, q1)
        irrelevances = gc._hi(q, q1)

        return fidelities, irrelevances
    
    def _compute_cdf(self, Z0, Z0i):
        """
        Internal method to compute CDF values using equation 15.29.
        
        Parameters
        ----------
        Z0 : ndarray
            Original domain evaluation points.
        Z0i : ndarray
            Transformed domain evaluation points.
            
        Returns
        -------
        ndarray
            Calculated CDF values at each evaluation point using EGDF formula.
        """
        # Handle empty data case
        if len(self.Zi) == 0:
            return np.zeros_like(Z0i)
        
        # Reshape for broadcasting
        Z0_w = Z0i.reshape(1, -1)  # Shape: (1, n_points)
        weights = self.weights.reshape(-1, 1)  # Shape: (n_samples, 1)
        
        # Calculate fidelities and irrelevances
        fidelities, irrelevances = self._calculate_fidelities_irrelevances(Z0_w)
        
        # Calculate weighted means using equation 15.31
        mean_fidelity = np.sum(weights * fidelities, axis=0) / np.sum(weights)  # f̄_E
        mean_irrelevance = np.sum(weights * irrelevances, axis=0) / np.sum(weights)  # h̄_E
        
        # Calculate estimating modulus using equation 15.28
        M_zi = np.sqrt(mean_fidelity**2 + mean_irrelevance**2)
        
        # Calculate EGDF using equation 15.29
        egdf_values = (1 - mean_irrelevance / M_zi) / 2
        
        return egdf_values
    
    def _compute_pdf(self, Z0, Z0i):
        """
        Internal method to compute PDF values using equations 15.30 and 15.31.
        
        Parameters
        ----------
        Z0 : ndarray
            Original domain evaluation points.
        Z0i : ndarray
            Transformed domain evaluation points.
            
        Returns
        -------
        ndarray
            Calculated PDF values at each evaluation point using EGDF density formula.
        """
        # Initialize output array
        density = np.zeros_like(Z0i, dtype=float)
        
        # Handle empty data case
        if len(self.Zi) == 0:
            return density
        
        # Handle zero and near-zero Z0 values
        eps = np.finfo(float).eps
        mask = np.abs(Z0i) > eps
        Z0_safe = Z0i[mask]
        
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
            density_safe = (1 / (self.sparam)) * (numerator / M_zi_cubed)
            
            # Assign calculated values to output array
            density[mask] = density_safe
            
            # Handle negative density values (as mentioned in the text, EGDF can have negative density)
            # We could either clip to zero or leave as is based on the application requirements
            if np.any(density < 0):
                warnings.warn("EGDF density contains negative values, which may indicate non-homogeneous data", RuntimeWarning)
        
        return density
    
    def cdf(self, Z0=None):
        """
        Compute the Estimating Global Distribution Function (EGDF).
        
        Parameters
        ---------- 
        Z0 : array-like, optional
            The reference points where to evaluate the distribution.
            
        Returns
        -------
        ndarray
            The estimated global distribution function values (CDF) at Z0.
        """
        # Check cache first
        if Z0 is not None and tuple(Z0) in self._cache:
            return self._cache[tuple(Z0)]['cdf']
        
        # Compute fit if not in cache
        result = self.fit(Z0, compute_pdf=False)
        return result['cdf']

    def density(self, Z0=None):
        """
        Calculate the EGDF probability density function.
        
        Parameters
        ----------
        Z0 : array-like, optional
            Grid points at which to evaluate the density.
            
        Returns
        -------
        ndarray
            Density values (PDF) for each point in Z0.
            
        Notes
        -----
        For homogeneous data, EGDF's density will be non-negative and have a single maximum.
        For non-homogeneous data, the density may become negative, signaling that EGDF
        is not appropriate for the dataset.
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
        Plot the EGDF and EGDF density on a single plot with dual y-axes.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.
        cdf : bool, optional
            If True, plot the cumulative distribution function (EGDF).
        pdf : bool, optional
            If True, plot the probability density function (EGDF density).
        Z0 : array-like, optional
            Evaluation points.
        n_points : int, optional
            Number of evaluation points if Z0 is None.
            
        Returns
        -------
        tuple or matplotlib.axes.Axes
            If pdf is True, returns (ax, ax2) - the primary and secondary axes.
            If pdf is False, returns just ax - the primary axis.
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
            line1, = ax.plot(Z0, cdf_values, 'b-', label='EGDF')
            ax.set_xlabel('Value')
            ax.set_ylabel('CDF', color='blue')
            ax.tick_params(axis='y', labelcolor='blue')
            ax.grid(True, linestyle='--', alpha=0.7)
            lines.append(line1)
            labels.append('EGDF (CDF)')
        
        # Create second y-axis and plot PDF
        if pdf:
            ax2 = ax.twinx()  # Create a second y-axis that shares the same x-axis
            pdf_values = result['pdf']
            line2, = ax2.plot(Z0, pdf_values, 'r-', label='EGDF Density')
            ax2.set_ylabel('Density', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            lines.append(line2)
            labels.append('EGDF Density (PDF)')
        else:
            ax2 = None
        
        # Add legend combining both plots
        if lines:
            ax.legend(lines, labels, loc='best')
        
        # Make sure both plots are visible
        fig = ax.figure
        fig.tight_layout()
        
        return ax if not pdf else (ax, ax2)
    
    def test_homogeneity(self):
        """
        Test the homogeneity of the data sample.
        
        The EGDF is suitable only for homogeneous data samples. This method tests
        whether the data sample is homogeneous by checking if the EGDF's density
        is non-negative over its full range and has only one maximum.
        
        Returns
        -------
        bool
            True if the data sample appears homogeneous (non-negative density with single maximum),
            False otherwise.
            
        Notes
        -----
        This test is based on the characteristics of EGDF described in the text:
        EGDF is suitable only for data samples with non-negative density and one maximum.
        """
        # Generate dense evaluation points
        Z0 = np.linspace(self.data_lb, self.data_ub, 200)
        
        # Get density values
        pdf_values = self.density(Z0)
        
        # Check if density is non-negative
        is_non_negative = np.all(pdf_values >= 0)
        
        # Check if density has only one maximum
        # Find all local maxima by comparing to neighboring values
        peaks = []
        for i in range(1, len(pdf_values) - 1):
            if pdf_values[i] > pdf_values[i-1] and pdf_values[i] > pdf_values[i+1]:
                peaks.append(i)
        
        has_single_max = len(peaks) <= 1
        
        # The data is homogeneous if density is non-negative and has at most one maximum
        return is_non_negative and has_single_max
        
    def get_scale_parameter(self):
        """
        Get the scale parameter used for the EGDF calculation.
        
        Returns
        -------
        float
            The scale parameter value. If S was set to 'auto', returns the 
            automatically calculated value.
        """
        return self.sparam