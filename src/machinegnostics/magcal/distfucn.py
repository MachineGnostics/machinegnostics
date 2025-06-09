import numpy as np
from machinegnostics.magcal.scale_param import ScaleParam
from machinegnostics.magcal.characteristics import GnosticsCharacteristics

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

        Example
        -------
        >>> gdf = GnosticDistributionFunction()
        >>> gdf.gnostic_kernel(1.0, 0.0, 1.0)
        0.5403023058681398
        """
        A = a - a0
        return 1.0 / (s * np.cosh(2 * A/s) ** 2)
    
    def gnostic_density(self, x, data, s=1.0, normalize=True):
        """
        Calculate the gnostic kernel density estimate at points x.
        
        Parameters:
            x      : array-like, points where to evaluate the density
            data   : array-like, data points (centers of kernels)
            s      : float or array-like, scale parameter(s) (can be scalar or same length as data)
            normalize : bool, if True, normalize so the area under density is 1
            
        Returns:
            density : array, density values at each x
        
        Example
        -------
        >>> gdf = GnosticDistributionFunction()
        >>> x = np.linspace(-5, 5, 100)
        >>> data = np.random.normal(0, 1, 50)
        >>> density = gdf.gnostic_density(x, data, s=1.0)
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
    
    def eldf(self, z, z0, s=1.0):
        '''
        ELDF - Estimating Local Distribution Function

        Parameters
        ---------- 
        z : array-like
            The data points for which to compute the EGDF.
        z0 : array-like
            The centers of the kernels (reference points).
        s : float, optional
            The scale parameter for the Gnostic kernel (default is 1.0).
        
        Returns
        -------
        el : array-like
            The estimated global distribution function values at the points `z`.

        Example
        -------
        >>> gdf = GnosticDistributionFunction()
        >>> z = np.random.normal(0, 1, 100)
        >>> z0 = np.linspace(-3, 3, 10)
        >>> el = gdf.egdf(z, z0, s=1.0)
        >>> print(el)
        '''
        Z = np.asarray(z)
        A0 = np.asarray(z0)
        el = np.zeros_like(A0, dtype=float)
        for Zk in Z:
            qk = (Zk / A0)**(2/S)
            el += 1 / (1 + qk**4)
        el /= len(Z)
        return el
    
    def _make_w(self, Z, W=None):
        """
        for internal use only.

        Calculates vector of equal weights with the sum normalized to 1,
        or checks the length and reshapes W to a column vector.
        
        Parameters:
        Z (array-like): Input vector.
        W (array-like, optional): Weight vector.

        Returns:
        numpy.ndarray: Column vector of weights.
        """
        Z = np.asarray(Z)
        
        if Z.ndim != 1:
            raise ValueError("Z must be a vector")
        
        if W is not None:
            W = np.asarray(W)
            if W.ndim != 1 or len(W) != len(Z):
                raise ValueError("W must be a vector of the same length as Z")
            W = W.reshape(-1, 1)
        else:
            W = np.ones(len(Z)).reshape(-1, 1) / len(Z)
        
        return W
    
    import numpy as np

    def _make_q(S, zk, Z, W=None, varS=0):
        """
        Calculates S and matrices of q and 1/q.
        Each row corresponds to zk for all Z.
        
        Parameters:
        S (float or array-like): Scale parameter (scalar or vector of same length as Z)
        zk (array-like): Scaling points (can be matrix or vector)
        Z (array-like): Data vector
        W (array-like, optional): Weight vector (should be column vector if used)
        varS (int, optional): If > 0 and S is scalar, compute variable scale using VarS
        
        Returns:
        S (float or ndarray): Updated scale parameter (scalar or column vector)
        q (ndarray): q matrix
        q1 (ndarray): 1/q matrix
        """
        scale = ScaleParam()
        Z = np.asarray(Z)
        zk = np.asarray(zk)
        
        if Z.ndim != 1:
            raise ValueError("Z must be a vector")
        
        if np.isscalar(S):
            is_scalar_S = True
        else:
            S = np.asarray(S)
            if S.ndim != 1 or len(S) != len(Z):
                raise ValueError("S must be a scalar or a vector of the same length as Z")
            is_scalar_S = False

        # Flatten zk into a column vector
        zk = zk.reshape(-1, 1)
        
        # Compute variable scale if requested
        if varS > 0 and is_scalar_S:
            if W is None:
                raise ValueError("W must be provided when varS > 0 and S is scalar")
            S = scale.var_s(Z, W, S)
        elif not is_scalar_S:
            S = S.reshape(-1, 1)

        # Calculate q and q1 if needed
        X = (1.0 / zk) @ Z.reshape(1, -1)
        gc = GnosticsCharacteristics(X)
        q, q1 = gc._get_q_q1(S=S)

        return S, q, q1
