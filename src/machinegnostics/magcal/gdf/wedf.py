from machinegnostics.magcal.util.logging import get_logger
import numpy as np
import logging

class WEDF:
    """
    Weighted Empirical Distribution Function (WEDF)
    
    This class implements the WEDF that accounts for data weights, which is useful
    when dealing with repeated values or data points of varying importance.
    """
    
    def __init__(self, data, weights=None, data_lb=None, data_ub=None, verbose=False):
        """
        Initialize the WEDF with data points and optional weights.
        
        Parameters
        ----------
        data : array-like
            Input data values
        weights : array-like, optional
            A priori weights for each data point. If None, equal weights are assigned.
        data_lb : float, optional
            Lower bound for the data range
        data_ub : float, optional
            Upper bound for the data range
        verbose : bool, optional
            If True, set logging level to DEBUG. Default is False.
        """
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if verbose else logging.WARNING)
        self.logger.debug(f"{self.__class__.__name__} initialized with parameters: %s", self.__dict__)

        # Convert inputs to numpy arrays and sort data
        self.data = np.asarray(data)
        if data_lb is None:
            self.data_lb = np.min(self.data)
        else:
            self.data_lb = data_lb
        if data_ub is None:
            self.data_ub = np.max(self.data)
        else:
            self.data_ub = data_ub
        if self.data_lb >= self.data_ub:
            self.logger.info("data_lb must be less than data_ub")
        if self.data.size == 0:
            self.logger.error("data must contain at least one element")
            raise ValueError("data must contain at least one element")
        if not np.issubdtype(self.data.dtype, np.number):
            self.logger.error("data must be numeric")
            raise ValueError("data must be numeric")
        
        # Sort data and corresponding weights
        sort_idx = np.argsort(self.data)
        self.data = self.data[sort_idx]
        
        if weights is None:
            # Equal weights if none provided
            self.weights = np.ones_like(self.data)
        else:
            weights = np.asarray(weights)
            self.weights = weights[sort_idx]
            
        # Normalize weights
        self.normalized_weights = self.weights / np.sum(self.weights)
        
        # Calculate WEDF values
        self._calculate_wedf()
    
    def _calculate_wedf(self):
        """Calculate the WEDF values at each data point."""
        n = len(self.data)
        self.wedf_values = np.zeros(n)
        
        # First value
        self.wedf_values[0] = self.normalized_weights[0] / 2
        
        # Remaining values using recursive relation
        for k in range(1, n):
            self.wedf_values[k] = (self.wedf_values[k-1] + 
                                  (self.normalized_weights[k-1] + self.normalized_weights[k]) / 2)
            
    def fit(self, z):
        """
        Fit the WEDF at given points.
        
        Parameters
        ----------
        z : float or array-like
            Points at which to fit the WEDF
        
        Returns
        -------
        float or ndarray
            WEDF values at the given points
        """
        self.logger.info("Fitting WEDF at given points.")
        z = np.asarray(z)
        single_value = z.ndim == 0
        
        if single_value:
            z = np.array([z])
            
        result = np.zeros_like(z, dtype=float)
        
        for i, point in enumerate(z):
            if point <= self.data[0]:
                result[i] = 0.0
            elif point >= self.data[-1]:
                result[i] = 1.0
            else:
                # Find the index of the largest data point less than z
                idx = np.searchsorted(self.data, point) - 1
                result[i] = self.wedf_values[idx]
        
        self.logger.info("WEDF fitting completed.")
        return result[0] if single_value else result
    
    def plot(self, ax=None):
        """
        Plot the WEDF.
        
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
            x = np.repeat(self.data, 2)[1:]
            y = np.repeat(self.wedf_values, 2)[:-1]
            
            # Add endpoints for proper step function
            x = np.concatenate([[self.data[0]], x, [self.data[-1]]])
            y = np.concatenate([[0], y, [1]])
            
            ax.plot(x, y, 'b-', label='WEDF')
            ax.set_xlabel('Data Value')
            ax.set_ylabel('Cumulative Probability')
            ax.set_title('Weighted Empirical Distribution Function')
            ax.grid(True)
            return ax
            
        except ImportError:
            self.logger.warning("Matplotlib is required for plotting.")
            return None
        
    def generate_ks_points(self, num_points=None):
        """
        Generate Kolmogorov-Smirnov points for distribution fitting.
        
        Parameters
        ----------
        num_points : int, optional
            Number of K-S points to generate. If None, uses the length of the data.
        Returns
        -------
        Z0 : ndarray
            Generated K-S points
        ks_probs : ndarray
            Corresponding probabilities for the K-S points
        """
        # Use data length if not specified
        L = num_points if num_points is not None else len(self.data)

        # Generate K-S probabilities
        ks_probs = np.arange(1, 2*L, 2) / (2*L)

        # Generate corresponding points
        data_range = self.data_ub - self.data_lb
        Z0 = self.data_lb + data_range * ks_probs

        return Z0, ks_probs