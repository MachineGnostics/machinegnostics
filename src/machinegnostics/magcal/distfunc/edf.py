"""
Machine Gnostics - Machine Gnostics Library
Copyright (C) 2025  Machine Gnostics Team

Author: Nirmal Parmar
License: GNU General Public License v3.0 (GPL-3.0)

# Gnostic Distribution Function Module

EDF (Empirical Distribution Function) is a non-parametric estimator of the cumulative distribution function (CDF) of a random variable.
"""

import numpy as np
from scipy.stats import kstest

class EDF:
    """
    Empirical Distribution Function (EDF)

    EDF is a non-parametric estimator of the cumulative distribution function (CDF) of a random variable.
    It is defined as the proportion of observations less than or equal to a given value.

    This implementation includes gnostic criterion functions for goodness-of-fit testing
    and supports comparison with other distribution functions (GDF, DDR).

    Attributes:
    - data: Input data for the distribution.
    - lb: Lower bound for the distribution.
    - ub: Upper bound for the distribution.
    """

    def __init__(self, data, lb=None, ub=None, weights=None):
        """
        Initialize the Empirical Distribution Function.
        
        Parameters
        ----------
        data : array-like
            Input data values
        lb : float, optional
            Lower bound for the data range
        ub : float, optional
            Upper bound for the data range
        weights : array-like, optional
            A priori weights for data points. If None, equal weights are assigned.
        """
        self.data = np.asarray(data)
        
        # Set bounds if not provided
        self.lb = lb if lb is not None else np.min(data) if data.size > 0 else 0
        self.ub = ub if ub is not None else np.max(data) if data.size > 0 else 1
        
        # Sort the data for EDF calculation
        self.sorted_data = np.sort(self.data)
        
        # Initialize weights if provided, otherwise use equal weights
        if weights is None:
            self.weights = np.ones_like(self.data)
        else:
            self.weights = np.asarray(weights)
            
        # Normalize weights
        self.normalized_weights = self.weights / np.sum(self.weights)
        
        # Calculate EDF values at each data point
        self._calculate_edf()
        
        # Initialize parameters for goodness-of-fit testing
        self.Z0 = None    # Evaluation points
        self.Z0_ref = None  # Reference ideal distribution
        self.S = 1.0      # Scale parameter
        
        # Cache for computed values (similar to ELDF)
        self._cache = {}
        
    def _calculate_edf(self):
        """Calculate the EDF values at each sorted data point."""
        n = len(self.sorted_data)
        if n == 0:
            self.edf_values = np.array([])
            return
            
        # For regular EDF (unweighted)
        if np.allclose(self.weights, self.weights[0]):
            # Standard EDF formula: i/n for each point
            self.edf_values = np.arange(1, n+1) / n
        else:
            # For weighted EDF, use cumulative sum of normalized weights
            sort_idx = np.argsort(self.data)
            sorted_weights = self.normalized_weights[sort_idx]
            self.edf_values = np.cumsum(sorted_weights)
    
    def fit(self, Z0=None, n_points=100, compute_pdf=True):
        """
        Compute EDF values at specified points.
        
        Parameters
        ----------
        Z0 : array-like, optional
            The reference points where to evaluate the distribution.
            If None, a linear space between lb and ub with n_points is used.
        n_points : int, optional
            Number of evaluation points if Z0 is None. Default is 100.
        compute_pdf : bool, optional
            Whether to compute PDF values (always returns zeros for EDF as it's a step function)
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'Z0': ndarray, evaluation points
            - 'cdf': ndarray, EDF values at evaluation points
            - 'pdf': ndarray, PDF approximation (if compute_pdf=True)
            
        Notes
        -----
        For EDF, the PDF is approximated as a simple step function derivative.
        """
        # Generate evaluation points if not provided
        if Z0 is None:
            self.Z0 = np.linspace(self.lb, self.ub, n_points)
        else:
            self.Z0 = np.asarray(Z0)
            
        # Check if result is already cached
        cache_key = tuple(self.Z0)
        if cache_key in self._cache:
            result = self._cache[cache_key]
            # If pdf was requested but not in cache, compute it
            if compute_pdf and 'pdf' not in result:
                result['pdf'] = self._compute_pdf(self.Z0)
            return result
        
        # Compute CDF values
        cdf_values = self.cdf(self.Z0)
        
        # Prepare result dictionary
        result = {
            'Z0': self.Z0,
            'cdf': cdf_values
        }
        
        # Compute PDF if requested
        if compute_pdf:
            pdf_values = self._compute_pdf(self.Z0)
            result['pdf'] = pdf_values
        
        # Cache results
        self._cache[cache_key] = result
        
        return result
    
    def cdf(self, x):
        """
        Evaluate the EDF (CDF) at given points.
        
        Parameters
        ----------
        x : float or array-like
            Points at which to evaluate the EDF
            
        Returns
        -------
        float or ndarray
            EDF values at the given points
        """
        x = np.asarray(x)
        single_value = x.ndim == 0
        
        if single_value:
            x = np.array([x])
            
        result = np.zeros_like(x, dtype=float)
        
        for i, point in enumerate(x):
            # For each point, count proportion of data points <= point
            if point < self.sorted_data[0]:
                result[i] = 0.0
            elif point >= self.sorted_data[-1]:
                result[i] = 1.0
            else:
                # Find the index of the largest data point less than or equal to x
                idx = np.searchsorted(self.sorted_data, point, side='right') - 1
                result[i] = self.edf_values[idx]
                
        return result[0] if single_value else result
    
    def _compute_pdf(self, x):
        """
        Approximate the PDF of the EDF at given points.
        
        For EDF, the PDF is a sum of Dirac delta functions at each data point,
        which we approximate as a histogram-like density.
        
        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the approximate PDF
            
        Returns
        -------
        ndarray
            Approximate PDF values (nonzero only at data points)
        """
        # Initialize with zeros
        pdf = np.zeros_like(x, dtype=float)
        
        # Calculate histogram bin width (use average spacing between sorted data)
        if len(self.sorted_data) > 1:
            avg_spacing = (self.sorted_data[-1] - self.sorted_data[0]) / (len(self.sorted_data) - 1)
            bin_width = avg_spacing
        else:
            bin_width = 1.0
        
        # For each data point, add contribution to nearby evaluation points
        for i, data_point in enumerate(self.sorted_data):
            # Find the closest evaluation point
            idx = np.abs(x - data_point).argmin()
            # Add contribution weighted by normalized weight
            weight = self.normalized_weights[np.where(self.data == data_point)[0][0]]
            pdf[idx] += weight / bin_width
        
        return pdf
    
    def density(self, x=None):
        """
        Calculate the approximate PDF of the EDF.
        
        Alias for compatibility with ELDF interface.
        
        Parameters
        ----------
        x : array-like, optional
            Points at which to evaluate the approximate PDF
            
        Returns
        -------
        ndarray
            Approximate PDF values
        
        Notes
        -----
        EDF is a step function with no proper density. This method
        returns an approximation with spikes at the data points.
        """
        if x is None:
            if self.Z0 is None:
                # Generate default points if none available
                x = np.linspace(self.lb, self.ub, 100)
            else:
                x = self.Z0
                
        # Check cache first
        cache_key = tuple(x)
        if cache_key in self._cache and 'pdf' in self._cache[cache_key]:
            return self._cache[cache_key]['pdf']
            
        # Compute PDF
        pdf_values = self._compute_pdf(x)
        
        # Store in cache if these are evaluation points
        if cache_key not in self._cache:
            self._cache[cache_key] = {'Z0': x, 'cdf': self.cdf(x)}
        self._cache[cache_key]['pdf'] = pdf_values
            
        return pdf_values
    
    def plot(self, ax=None, cdf=True, pdf=False, Z0=None, n_points=100):
        """
        Plot the EDF and optionally its approximate density.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
        cdf : bool, optional
            If True, plot the cumulative distribution function (EDF). Default is True.
        pdf : bool, optional
            If True, plot the approximate probability density function. Default is False.
        Z0 : array-like, optional
            Evaluation points. If None, a linear space is used.
        n_points : int, optional
            Number of evaluation points if Z0 is None. Default is 100.
            
        Returns
        -------
        tuple or matplotlib.axes.Axes
            If pdf is True, returns (ax, ax2) - the primary and secondary axes.
            If pdf is False, returns just ax - the primary axis.
        """
        try:
            import matplotlib.pyplot as plt
            if ax is None:
                fig, ax = plt.subplots()
                
            # Compute fit at evaluation points
            result = self.fit(Z0, n_points, compute_pdf=pdf)
            Z0 = result['Z0']
            
            lines = []
            labels = []
            
            if cdf:
                # Create a step function representation
                x = np.repeat(self.sorted_data, 2)[1:]
                y = np.repeat(self.edf_values, 2)[:-1]
                
                # Add endpoints for proper step function
                x = np.concatenate([[self.sorted_data[0]], x, [self.sorted_data[-1]]])
                y = np.concatenate([[0], y, [1]])
                
                line1, = ax.plot(x, y, 'b-', label='EDF')
                ax.set_xlabel('Value')
                ax.set_ylabel('Cumulative Probability', color='blue')
                ax.tick_params(axis='y', labelcolor='blue')
                ax.grid(True, alpha=0.3)
                
                lines.append(line1)
                labels.append('EDF (CDF)')
            
            # Create second y-axis for PDF if requested
            if pdf:
                ax2 = ax.twinx()
                pdf_values = result['pdf']
                
                # FIX: Don't try to unpack the stem container
                stem_container = ax2.stem(self.sorted_data, 
                            [pdf_values[np.abs(Z0 - point).argmin()] for point in self.sorted_data],
                            'r-', markerfmt='ro', basefmt=' ', label='Empirical Density')
                
                ax2.set_ylabel('Density', color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                
                # Use markerline for legend
                lines.append(stem_container.markerline)
                labels.append('EDF Density (PDF)')
            else:
                ax2 = None
            
            # Add legend combining both plots
            if lines:
                ax.legend(lines, labels, loc='best')
                
            # Adjust layout
            fig = ax.figure
            fig.tight_layout()
            
            return ax if not pdf else (ax, ax2)
            
        except ImportError:
            print("Matplotlib is required for plotting.")
            return None

    def evaluate(self, x):
        """
        Legacy method for backward compatibility. Use cdf() instead.
        
        Parameters
        ----------
        x : float or array-like
            Points at which to evaluate the EDF
            
        Returns
        -------
        float or ndarray
            EDF values at the given points
        """
        return self.cdf(x)
    
    def set_ideal_distribution(self, Z0, S=1.0):
        """
        Set the ideal distribution value and scale parameter for goodness-of-fit testing.
        
        Parameters
        ----------
        Z0 : float or callable
            Ideal data value or a function that evaluates the ideal GDF
        S : float, optional
            Scale parameter controlling the curvature of the distribution
        """
        self.Z0_ref = Z0
        self.S = S
    
    def calculate_fidelity(self, GDF, DDR=None):
        """
        Calculate fidelity (posterior weights) for goodness-of-fit.
        
        Parameters
        ----------
        GDF : callable
            Gnostic Distribution Function to compare with
        DDR : callable, optional
            Data Distribution Reference (defaults to this EDF if None)
            
        Returns
        -------
        tuple
            (fidelity, irrelevance) values for each data point
        """
        # If no DDR provided, use this EDF as reference
        if DDR is None:
            DDR = self.cdf
        
        # Calculate q values using equation 15.14: q_k = (GDF(Z_k)/DDR(Z_k))^(1/S)
        gdf_values = np.asarray([GDF(z) for z in self.sorted_data])
        ddr_values = np.asarray([DDR(z) for z in self.sorted_data])
        
        # Avoid division by zero
        valid_idx = ddr_values > 0
        q_values = np.zeros_like(gdf_values, dtype=float)
        q_values[valid_idx] = (gdf_values[valid_idx] / ddr_values[valid_idx]) ** (1/self.S)
        
        # Calculate fidelity using equation 15.12: f_E,k = 2/(q²_k + 1/q²_k)
        q_squared = q_values**2
        fidelity = np.zeros_like(q_squared)
        valid_q = q_squared > 0
        fidelity[valid_q] = 2 / (q_squared[valid_q] + 1/q_squared[valid_q])
        
        # Calculate irrelevance using equation 15.12: h_E,k = (q²_k - 1/q²_k)/(q²_k + 1/q²_k)
        irrelevance = np.zeros_like(q_squared)
        irrelevance[valid_q] = (q_squared[valid_q] - 1/q_squared[valid_q]) / (q_squared[valid_q] + 1/q_squared[valid_q])
        
        return fidelity, irrelevance
    
    def criterion_functions(self, GDF, DDR=None):
        """
        Calculate criterion functions for goodness-of-fit assessment.
        
        Parameters
        ----------
        GDF : callable
            Gnostic Distribution Function to compare with
        DDR : callable, optional
            Data Distribution Reference (defaults to this EDF if None)
            
        Returns
        -------
        dict
            Dictionary of criterion function values:
            - CF_f: Mean fidelity (eq 15.16)
            - CF_h2: Mean squared irrelevance (eq 15.17)
            - CF_I: Entropy criterion (eq 15.18)
        """
        # Calculate fidelity and irrelevance
        fidelity, irrelevance = self.calculate_fidelity(GDF, DDR)
        
        # Calculate criterion functions using equations 15.16-15.18
        CF_f = np.mean(fidelity)  # eq 15.16: CF(f_E) := mean(f_E)
        CF_h2 = np.mean(irrelevance**2)  # eq 15.17: CF(h²_E) := mean(h²_E)
        
        # Calculate p_k values: p_k = (1 - h_E,k)/2
        p_values = (1 - irrelevance) / 2
        
        # Calculate entropy criterion: CF(I) := -p*ln(p) (eq 15.18)
        # Avoid ln(0) issues
        valid_p = p_values > 0
        CF_I = 0
        if np.any(valid_p):
            CF_I = -np.sum(p_values[valid_p] * np.log(p_values[valid_p]))
        
        return {
            'CF_f': CF_f,
            'CF_h2': CF_h2,
            'CF_I': CF_I
        }
    
    def ks_test(self, GDF):
        """
        Perform Kolmogorov-Smirnov test against the given distribution.
        
        Parameters
        ----------
        GDF : callable
            Distribution function to test against
            
        Returns
        -------
        tuple
            (statistic, p_value) from the KS test
        """
        # Get sorted data
        data = self.sorted_data
        
        # Evaluate GDF at each data point
        cdf_values = np.array([GDF(x) for x in data])
        
        # Calculate KS statistic directly
        n = len(data)
        edf_values = np.arange(1, n+1) / n
        ks_stat = np.max(np.abs(edf_values - cdf_values))
        
        # Use scipy's function to get the p-value
        _, p_value = kstest(data, GDF)
        
        return ks_stat, p_value
    
    def generate_ks_points(self, num_points=None):
        """
        Generate Kolmogorov-Smirnov points for distribution fitting.
        
        Parameters
        ----------
        num_points : int, optional
            Number of K-S points to generate. If None, uses the length of the data.
            
        Returns
        -------
        tuple
            (Z0, ks_probs) - Generated K-S points and their probabilities
        """
        # Use data length if not specified
        L = num_points if num_points is not None else len(self.data)
        
        # Generate K-S probabilities: (2k-1)/(2L) for k=1,2,...,L
        ks_probs = np.arange(1, 2*L, 2) / (2*L)
        
        # Generate corresponding points in the data range
        data_range = self.ub - self.lb
        Z0 = self.lb + data_range * ks_probs
        
        return Z0, ks_probs