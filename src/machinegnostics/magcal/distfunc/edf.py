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
        self.lb = lb
        self.ub = ub
        
        # Sort the data for EDF calculation
        self.sorted_data = np.sort(self.data)
        
        # Set bounds if not provided
        if self.lb is None:
            self.lb = np.min(data) if data.size > 0 else 0
        if self.ub is None:
            self.ub = np.max(data) if data.size > 0 else 1
            
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
        self.Z0 = None  # Ideal data value (GDF reference)
        self.S = 1.0    # Scale parameter
        
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
    
    def evaluate(self, x):
        """
        Evaluate the EDF at given points.
        
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
    
    def plot(self, ax=None):
        """
        Plot the EDF.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
            
        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot
        """
        try:
            import matplotlib.pyplot as plt
            if ax is None:
                fig, ax = plt.subplots()
                
            # Create a step function representation
            x = np.repeat(self.sorted_data, 2)[1:]
            y = np.repeat(self.edf_values, 2)[:-1]
            
            # Add endpoints for proper step function
            x = np.concatenate([[self.sorted_data[0]], x, [self.sorted_data[-1]]])
            y = np.concatenate([[0], y, [1]])
            
            ax.plot(x, y, 'b-', label='EDF')
            ax.set_xlabel('Data Value')
            ax.set_ylabel('Cumulative Probability')
            ax.set_title('Empirical Distribution Function')
            ax.grid(True)
            return ax
            
        except ImportError:
            print("Matplotlib is required for plotting.")
            return None

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
        self.Z0 = Z0
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
            DDR = self.evaluate
        
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