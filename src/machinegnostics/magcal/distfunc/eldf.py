"""
Machine Gnostics - Machine Gnostics Library
Copyright (C) 2025  Machine Gnostics Team

Author: Nirmal Parmar
License: GNU General Public License v3.0 (GPL-3.0)

# Gnostic Distribution Function Module

ELDF - Estimating Local Distribution Function
"""

import numpy as np
import warnings
from machinegnostics.magcal.distfunc.data_transform import DataDomainTransformation
from machinegnostics.magcal.scale_param import ScaleParam
from machinegnostics.magcal.mg_weights import GnosticsWeights
from machinegnostics.magcal.characteristics import GnosticsCharacteristics

class ELDF:
    """
    Estimating Local Distribution Function (ELDF)
    
    A gnostic approach to probability distribution estimation that employs
    locality-based weighting for robust statistical analysis. ELDF provides
    both cumulative distribution function (CDF) and probability density 
    function (PDF) estimations based on input data.
    
    ELDF is obtained as an arithmetical mean of gnostic kernels provided with their
    individual or joint scale parameters. The scale parameter controls the smoothness
    of the kernels, offering unique flexibility to reveal detailed data structure,
    isolate outliers, and identify sub-samples or clusters within the data.
    
    The number of visible clusters in the data depends on the choice of scale parameter (S).
    Smaller S values (e.g., S=0.1) reveal individual sharp clusters for each distinct data value,
    while larger values produce smoother distributions that may merge clusters. This ability
    to adjust cluster resolution by varying S makes ELDF particularly valuable for exploring
    data structure and identifying potential underlying processes or patterns.
    
    This unlimited flexibility is unique to ELDF and not shared with other distribution functions,
    making it an excellent tool for data analysis, visualization of data structure, and marginal
    analysis (decomposing samples by isolating data belonging to individual clusters).
    
    The ELDF method is particularly effective for handling datasets with 
    outliers or irregular distributions by using a scale parameter (S) that 
    controls the locality sensitivity of the estimation.
    
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
    varS : bool
        Whether scale parameter S varies per data point
    S : float or ndarray or str
        Scale parameter(s) or 'auto' for automatic calculation
    sparam : float or ndarray
        The computed scale parameter(s) after fitting
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
                 varS:bool=False,
                 homogeneous:bool=True):
        """
        Initialize the ELDF with data points and optional weights.
        
        Parameters
        ----------
        data : array-like
            Input data values for distribution estimation.
        weights : array-like, optional
            Weights for each data point. If None, equal weights are used.
        S : float or array-like or str, optional
            Smoothing parameter(s) controlling the locality sensitivity. 
            If 'auto', S will be calculated automatically based on the data.
            If varS is True, S can be an array-like of the same length as data.
            Recommended range is (0, 2] for numerical stability.
            Lower S values (e.g., 0.1) reveal individual clusters for each distinct data value,
            while higher values produce smoother distributions that may merge clusters.
        data_form : str, optional
            Data form specification:
            - 'a' for additive (default)
            - 'm' for multiplicative
            - None for no transformation
        data_lb : float, optional
            Lower bound for the data range. If None, min(data) is used.
        data_ub : float, optional
            Upper bound for the data range. If None, max(data) is used.
        varS : bool, optional
            If True, allows S to vary per data point. In this case, S should be 
            an array-like of the same length as data, or 'auto' for automatic calculation.
        homogeneous : bool, optional
            If True (default), uses the provided weights directly.
            If False, calculates homogeneous gnostic weights based on the data,
            which can provide more robust estimation for non-homogeneous distributions.

        Examples
        --------
        >>> eldf = ELDF(data, weights=weights, S=1.0, varS=True)
        >>> eldf.fit(n_points=200)
        >>> cdf_values = eldf.cdf()  # Get CDF values
        >>> pdf_values = eldf.density()  # Get PDF values
        >>> eldf.plot()  # Plot both CDF and PDF
        >>> eldf.plot(cdf=True, pdf=True)  # Plot both CDF and PDF on dual y-axes
        >>> eldf.plot(cdf=True, pdf=False)  # Plot only CDF
        >>> eldf.plot(cdf=False, pdf=True)  # Plot only PDF

        Raises
        ------
        ValueError
            If weights and data have different lengths, if data bounds are invalid,
            if data is empty or non-numeric, if S has invalid format based on varS setting,
            or if homogeneous is not a boolean value.
            
        Notes
        -----
        - The weights are normalized to sum to the number of data points.
        - When varS is True and S is 'auto', the scale parameters are calculated 
          automatically for each data point using the ScaleParam class.
        - S values > 2 may lead to numerical instability.
        - Experimenting with different S values can reveal interesting data structures
          and clusters that may correspond to underlying micro-processes in the data.
        - When homogeneous=False, the GnosticsWeights class is used to calculate
          weights that better handle non-homogeneous data distributions.
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
        self.Zi = self.transformer.transform_input(self.data)
        
        # Default evaluation points will be set in fit()
        self.Z0 = None
        self.Z0i = None

        # if homogeneous, use weights as is
        if self.homogeneous:
            self.weights = self.weights
        else:
            gw = GnosticsWeights()
            homogeneous_weights = gw._get_gnostic_weights(self.Z)
            self.weights = homogeneous_weights

        # varS and S handling
        self.scale = ScaleParam()
        self.varS = varS
        self.S = S
        if self.varS:
            if not isinstance(self.S, (list, np.ndarray, str)):
                raise ValueError("If varS is True, S must be an array-like of the same length as data or 'auto'")
            if isinstance(self.S, (list, np.ndarray)):
                if len(self.S) != len(self.data):
                    raise ValueError("S must have the same length as data when varS is True or 'auto'")
                self.S = np.asarray(self.S, dtype=float)
                if np.any(self.S <= 0) or np.any(self.S > 2):
                    warnings.warn("S values > 2 may lead to numerical instability", RuntimeWarning)
            if isinstance(self.S, str):
                self.S = self.S.lower()
                if self.S != 'auto':
                    raise ValueError("If varS is True, S must be an array-like of the same length as data or 'auto'")
        else:
            # If varS is False, ensure S is a single value
            if not isinstance(self.S, (int, float, str)):
                raise ValueError("If varS is False, S must be a single numeric value or 'auto'")
            if isinstance(self.S, str):
                self.S = self.S.lower()
                if self.S != 'auto':
                    raise ValueError("If varS is False, S must be a single numeric value or 'auto'")
            if isinstance(self.S, (int, float)):
                self.S = float(self.S)
                if self.S <= 0 or self.S > 2:
                    warnings.warn("S must be in the range (0, 2] for stability when varS is False", RuntimeWarning)

        # Cache for computed values
        self._cache = {}
    
    def fit(self, Z0=None, n_points=100, compute_pdf=True):
        """
        Compute both ELDF (CDF) and optionally PDF at specified points.
        
        This method evaluates the estimated distribution at given points, calculating
        both cumulative distribution and density values. Results are cached for 
        efficient repeated access.
        
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
            - 'S': float or ndarray, the scale parameter(s) used in the calculation

        Examples
        --------
        >>> eldf = ELDF(data, weights=weights, S=1.0, varS=True)
        >>> result = eldf.fit(n_points=200)
        >>> print(result['Z0'])  # Evaluation points
        >>> print(result['cdf'])  # CDF values at evaluation points
        >>> print(result['pdf'])  # PDF values at evaluation points (if compute_pdf=True)
        Notes
        -----
        - This method caches results for each unique set of evaluation points.
        - The internal calculations use transformed data based on the data_form.
        - The CDF is constructed as an arithmetical mean of gnostic kernels,
          each with its own scale parameter when varS=True.
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
        
        # Prepare result dictionary
        result = {
            'Z0': self.Z0,
            'cdf': cdf_values,
            'S': self.sparam
        }
        
        # Compute PDF if requested
        if compute_pdf:
            pdf_values = self._compute_pdf(self.Z0, self.Z0i)
            # transform PDF values to original domain
            # pdf_values = self.transformer.transform_output(pdf_values)
            result['pdf'] = pdf_values
        
        # Cache results
        self._cache[tuple(self.Z0)] = result
        
        return result
    
    def _compute_cdf(self, Z0, Z0i):
        """
        Internal method to compute CDF values.
        
        Calculates the Estimating Local Distribution Function values at the given
        evaluation points using either constant or variable scale parameters.
        
        Parameters
        ----------
        Z0 : ndarray
            Original domain evaluation points.
        Z0i : ndarray
            Transformed domain evaluation points.
            
        Returns
        -------
        ndarray
            Calculated CDF values at each evaluation point.
            
        Notes
        -----
        - Uses broadcasting to efficiently calculate values for all data points.
        - Handles the scale parameter S differently based on whether varS is True.
        - Small epsilon values are used to prevent division by zero.
        - The final CDF values are normalized by the sum of weights.
        - The calculation uses gnostic kernels whose shape depends on the scale parameter,
          allowing the CDF to adapt to the structure of the data.
        """
        # Reshape data for broadcasting
        Z_working = self.Zi.reshape(-1, 1)     # Shape: (n_samples, 1)
        Z0_w = Z0i.reshape(1, -1)              # Shape: (1, n_points)
        weights = self.weights.reshape(-1, 1)  # Shape: (n_samples, 1)
        
        # Handle division-by-zero
        eps = np.finfo(float).eps
        Z0_safe = np.maximum(Z0_w, eps)

        # calculation with VarS
        if self.varS:
            if self.S == 'auto':
                # Automatically calculate S based on the scale parameter
                S = self.scale.var_s(Z=Z_working, W=weights, S=1)
            else:
                # Use provided S values directly
                S = self.S
            # Reshape S for broadcasting
            S = S.reshape(-1, 1)
            # Ensure S is a scalar for broadcasting   
            self.sparam = S 

            gc = GnosticsCharacteristics(R = (Z_working / Z0_w))
            q, q1 = gc._get_q_q1(S=S)
            
            # Calculate ratio in working domain
            # qk = (Z_working / Z0_safe) ** (1 / S)   
            # Apply weights to the calculation
            eldf_values = np.sum(weights * (1 / (1 + q**4)), axis=0) / np.sum(weights)
        else:
            if self.S == 'auto':
                # Automatically calculate S based on the scale parameter
                S = np.mean(self.scale.var_s(Z=Z_working, W=weights, S=1))
            else:
                # Use provided S values directly
                S = self.S
            # Ensure S is a scalar for broadcasting   
            self.sparam = S 

            gc = GnosticsCharacteristics(R = (Z_working / Z0_w))
            q, q1 = gc._get_q_q1(S=S)

            # Calculate ratio in working domain
            # qk = (Z_working / Z0_safe) ** (1 / S)
            
            # Apply weights to the calculation
            eldf_values = np.sum(weights * (1 / (1 + q**4)), axis=0) / np.sum(weights)
            
        # # Apply boundary corrections
        # min_data = np.min(self.data)
        # max_data = np.max(self.data)
        
        # # Find where evaluation points are beyond data range
        # below_min = Z0 < min_data
        # above_max = Z0 > max_data
        
        # # Handle points below minimum (set to 0 or small epsilon)
        # if np.any(below_min):
        #     eldf_values[below_min] = 0.0
        
        # # Handle points above maximum (set to 1)
        # if np.any(above_max):
        #     # Find the CDF value at the maximum data point
        #     # (or use the maximum CDF value within the data range)
        #     max_cdf = np.max(eldf_values[~above_max])
        #     eldf_values[above_max] = max_cdf

        return eldf_values
    
    def _compute_pdf(self, Z0, Z0i):
        """
        Internal method to compute PDF values.
        
        Calculates the probability density function values at given evaluation
        points using either constant or variable scale parameters.
        
        Parameters
        ----------
        Z0 : ndarray
            Original domain evaluation points.
        Z0i : ndarray
            Transformed domain evaluation points.
            
        Returns
        -------
        ndarray
            Calculated PDF values at each evaluation point.
            
        Notes
        -----
        - Handles edge cases such as empty data or zero/near-zero Z0 values.
        - Uses broadcasting for efficient calculation across all data points.
        - The density calculation uses different formulas based on whether varS is True.
        - Includes safeguards against division by zero with small epsilon values.
        - Returns an array of zeros for points where calculation is not possible.
        - The PDF may reveal multiple clusters depending on the scale parameter value,
          with smaller S values highlighting more detailed structure in the data.
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
        
        if len(Z0_safe) > 0:  # Only proceed if we have valid Z0 values
            # Reshape data for broadcasting
            Z_working = self.Zi.reshape(-1, 1)          # Shape: (n_samples, 1)
            Z0_w = Z0_safe.reshape(1, -1)               # Shape: (1, n_points)
            weights = self.weights.reshape(-1, 1)        # Shape: (n_samples, 1)
            
            # calculate with VarS
            if self.varS:
                if self.S == 'auto':
                    # Automatically calculate S based on the scale parameter
                    S = self.scale.var_s(Z=Z_working, W=weights, S=1)
                else:
                    # Use provided S values directly
                    S = self.S
                # Reshape S for broadcasting
                S = S.reshape(-1, 1)
                # Ensure S is a scalar for broadcasting   
                self.sparam = S 

                # Calculate q-matrix using the correct formula in working domain
                gc = GnosticsCharacteristics(R = (Z_working / Z0_w))
                q, q1 = gc._get_q_q1(S=S)
                # Calculate denominator
                denom = (q + q1)**2

                # Calculate result matrix, handling potential division by zero
                valid_denom = denom > eps
                result = np.zeros_like(denom, dtype=float)
                result[valid_denom] = 4 / denom[valid_denom]
                
                # Apply weights and scale by S - handle 2D result
                weighted_result = weights * result
                # Sum over data points (axis 0) to get 1D result
                density_safe = np.sum(weighted_result, axis=0) / (np.sum(weights) * np.mean(S))
                                
            else:
                # Varying S values need more complex broadcasting
                if self.S == 'auto':
                    # Automatically calculate S based on the scale parameter
                    S = np.mean(self.scale.var_s(Z=Z_working, W=weights, S=1))
                else:
                    # Use provided S values directly
                    S = self.S
                # Ensure S is a scalar for broadcasting   
                self.sparam = S          
                
                # Calculate q-matrix using the correct formula in working domain
                gc = GnosticsCharacteristics(R = (Z_working / Z0_w))
                q, q1 = gc._get_q_q1(S=S)
                # Calculate denominator
                denom = (q + q1)**2
                
                # Calculate contribution per point
                valid_denom = denom > eps
                result = np.zeros_like(denom, dtype=float)
                
                # Broadcasting S for division
                S_expanded = np.broadcast_to(S, denom.shape)
                
                # Calculate where denominator is valid
                result[valid_denom] = 4 / (S_expanded[valid_denom] * denom[valid_denom])
                
                # Apply weights to get the final density
                density_safe = np.sum(weights * result, axis=0) / np.sum(weights)
            
            # Ensure density_safe is 1D before assignment
            if len(density_safe.shape) > 1:
                density_safe = np.squeeze(density_safe)
            
            # Now assign to density array
            density[mask] = density_safe
        
        return density
    
    def cdf(self, Z0=None):
        """
        Compute the Estimating Local Distribution Function (ELDF).
        
        Provides the cumulative distribution function values at specified points,
        using cached results when available.
        
        Parameters
        ---------- 
        Z0 : array-like, optional
            The reference points where to evaluate the distribution.
            If None, uses the points from the last fit() call or generates
            a default set of points.
        
        Returns
        -------
        ndarray
            The estimated local distribution function values (CDF) at Z0.

        Examples
        --------
        >>> eldf = ELDF(data, weights=weights, S=1.0, varS=True)
        >>> cdf_values = eldf.cdf()  # Get CDF values at default points
        >>> cdf_values = eldf.cdf(Z0=np.linspace(0, 10, 100))  # Get CDF values at specified points
        >>> print(cdf_values)  # Print the computed CDF values

        Notes
        -----
        - Checks the cache first before computing new values.
        - If Z0 is not provided and no prior fit exists, a default fit will be performed.
        - This method provides a convenient interface to access CDF values without
          explicitly calling fit().
        - The CDF is constructed as an arithmetic mean of gnostic kernels,
          providing flexibility to reveal detailed data structure.
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
        
        Provides the probability density function values at specified points,
        using cached results when available. The PDF shows the relative likelihood
        of the data taking on given values, and can reveal clusters in the data
        depending on the scale parameter S.
        
        Parameters
        ----------
        Z0 : array-like, optional
            Grid points at which to evaluate the density.
            If None, uses the points from the last fit() call or generates
            a default set of points.
            
        Returns
        -------
        ndarray
            Density values (PDF) for each point in Z0.
            
        Notes
        -----
        - Checks the cache first before computing new values.
        - If PDF values were not computed in a previous call for the same Z0,
          they will be computed now.
        - If Z0 is not provided and no prior fit exists, a default fit will be performed.
        - This method provides a convenient interface to access PDF values without
          explicitly calling fit().
        - The PDF may reveal multiple clusters depending on the scale parameter value,
          with smaller S values highlighting more detailed structure in the data.
        - The number of clusters visible in the PDF depends on the choice of scale parameter,
          making it useful for identifying potential subgroups or patterns in the data.
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
        
        Creates a visualization of the estimated distribution, showing CDF on the
        left y-axis and PDF on the right y-axis. This visualization is particularly
        useful for examining the data structure, identifying clusters, and
        analyzing the effects of different scale parameters.
    
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
        cdf : bool, optional
            If True, plot the cumulative distribution function (ELDF). Default is True.
        pdf : bool, optional
            If True, plot the probability density function (ELDF density). Default is True.
        Z0 : array-like, optional
            Evaluation points. If None, a linear space is used.
        n_points : int, optional
            Number of evaluation points if Z0 is None. Default is 100.
            
        Returns
        -------
        tuple or matplotlib.axes.Axes
            If pdf is True, returns (ax, ax2) - the primary and secondary axes.
            If pdf is False, returns just ax - the primary axis.
            
        Notes
        -----
        - Uses matplotlib for visualization.
        - CDF is plotted in blue on the left y-axis.
        - PDF is plotted in red on the right y-axis (if pdf=True).
        - The plot includes appropriate labels, colors, and a legend.
        - The figure layout is automatically adjusted for readability.
        - The PDF plot may reveal clusters depending on the scale parameter,
          making it valuable for data structure visualization and cluster identification.
        - Comparing plots with different scale parameters can provide insights
          into the hierarchical structure of the data.
        
        Requires
        --------
        matplotlib
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
            line1, = ax.plot(Z0, cdf_values, 'b-', label='ELDF')
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
            line2, = ax2.plot(Z0, pdf_values, 'r-', label='ELDF Density')
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
    
    def gnostic_mean(self):
        """
        Estimate the gnostic mean of the distribution.
        
        Finds the mode of the probability density function, which is the value
        where the PDF reaches its maximum. In gnostic analysis, this represents
        the most typical or representative value in the data distribution.
        
        Returns
        -------
        float
            The x-value where the probability density function is maximized.
            
        Notes
        -----
        - Unlike the traditional mean (expected value), the gnostic mean
          represents the most probable value in the distribution.
        - This is equivalent to finding the mode of the distribution.
        - For multimodal distributions (common with small S values in ELDF),
          this returns the location of the highest peak.
        
        Examples
        --------
        >>> eldf = ELDF(data, S=1.0)
        >>> eldf.fit()
        >>> gnostic_mean = eldf.gnostic_mean()
        >>> print(f"Most representative value: {gnostic_mean}")
        """
        # Find the mode (maximum density) of the distribution
        pdf_values = self.density(self.Z0)
        mode = self.Z0[np.argmax(pdf_values)]
        return mode
    
    def get_bounds(self):
        """
        Get locations of minimum and maximum density in the distribution.
        
        Finds the x-values where the probability density function reaches
        its minimum and maximum values, representing the least and most
        likely values in the distribution.
        
        Returns
        -------
        tuple
            A tuple containing (lower_bound, upper_bound), where:
            - lower_bound: x-value where the PDF is minimized
            - upper_bound: x-value where the PDF is maximized
            
        Notes
        -----
        - The lower_bound represents the value with the lowest probability density
        - The upper_bound is identical to the value returned by gnostic_mean()
        - These bounds are not the support of the distribution, but rather
          the points of minimum and maximum likelihood
        - For distributions with multiple local minima or maxima, this returns
          the global extrema
          
        Examples
        --------
        >>> eldf = ELDF(data, S=1.0)
        >>> eldf.fit()
        >>> min_density_point, max_density_point = eldf.get_bounds()
        """
        pdf_values = self.density(self.Z0)
        lower_bound = self.Z0[np.argmin(pdf_values)]
        upper_bound = self.Z0[np.argmax(pdf_values)]
        return lower_bound, upper_bound

    # def mean(self):
    #     """Calculate the mean of the distribution"""
    #     Z0 = np.linspace(self.data_lb, self.data_ub, 1000)
    #     pdf_values = self.density(Z0)
    #     dx = Z0[1] - Z0[0]
    #     return np.sum(Z0 * pdf_values) * dx
    
    # def variance(self):
    #     """Calculate the variance of the distribution"""
    #     Z0 = np.linspace(self.data_lb, self.data_ub, 1000)
    #     pdf_values = self.density(Z0)
    #     dx = Z0[1] - Z0[0]
    #     mean = self.mean()
    #     return np.sum((Z0 - mean)**2 * pdf_values) * dx
    
    # def skewness(self):
    #     """Calculate the skewness of the distribution"""
    #     Z0 = np.linspace(self.data_lb, self.data_ub, 1000)
    #     pdf_values = self.density(Z0)
    #     dx = Z0[1] - Z0[0]
    #     mean = self.mean()
    #     var = self.variance()
    #     std = np.sqrt(var)
    #     return np.sum(((Z0 - mean) / std)**3 * pdf_values) * dx
    
    # def kurtosis(self):
    #     """Calculate the excess kurtosis of the distribution"""
    #     Z0 = np.linspace(self.data_lb, self.data_ub, 1000)
    #     pdf_values = self.density(Z0)
    #     dx = Z0[1] - Z0[0]
    #     mean = self.mean()
    #     var = self.variance()
    #     std = np.sqrt(var)
    #     return np.sum(((Z0 - mean) / std)**4 * pdf_values) * dx - 3
    
