import numpy as np


class GnosticDistributionFunction:
    """
    Base class for Gnostic distribution functions.
    
    This class provides a template for implementing Gnostic distribution functions
    with methods for calculating the G-median and other related statistics.
    """
    
    def __init__(self, tol=1e-6):
        """
        Initialize the GnosticDistributionFunction with data and tolerance.
        
        Parameters
        ----------
        tol : float, optional
            Tolerance for convergence (default is 1e-6).
        """
        self.tol = tol

    def gnostic_kernel(self, a, a0, s):
        """
        Gnostic kernel function.
        
        Parameters
        ----------
        a : float
            The value to evaluate the kernel at.
        a0 : float
            The center of the kernel.
        s : float
            The scale parameter of the kernel.
        
        Returns
        -------
        float
            The value of the Gnostic kernel at `a`.
        """
        return 1.0 / (np.cosh(2 * (a - a0)/s) ** 2)
    
    def gnostic_density(x, data, s=1.0, normalize=True):
        """
        Calculate the gnostic kernel density estimate at points x.
        
        Parameters:
            x      : array-like, points where to evaluate the density
            data   : array-like, data points (centers of kernels)
            s      : float or array-like, scale parameter(s) (can be scalar or same length as data)
            normalize : bool, if True, normalize so the area under density is 1
            
        Returns:
            density : array, density values at each x
        """
        x = np.asarray(x)
        data = np.asarray(data)
        if np.isscalar(s):
            s = np.full_like(data, s, dtype=float)
        else:
            s = np.asarray(s)
            assert len(s) == len(data), "Scale array must match data length"
        
        density = np.zeros_like(x, dtype=float)
        for xi, si in zip(data, s):
            density += self.gnostic_kernel(x, xi, si)
        density /= (len(data)*2)  # mean kernel (ELDF/EGDF style)
        
        if normalize:
            area = np.trapz(density, x)
            if area > 0:
                density /= area
        return density