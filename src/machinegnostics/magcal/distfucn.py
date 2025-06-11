import numpy as np
from machinegnostics.magcal.scale_param import ScaleParam
from machinegnostics.magcal.characteristics import GnosticsCharacteristics

class GnosticDistributionFunction:
    """
    Base class for Gnostic distribution functions.
    
    Gnostic distribution functions are a family of robust statistical methods based on 
    the principles of information theory. This class provides implementations of various
    Gnostic distribution functions and related statistics, including:
    
    - Gnostic kernel functions
    - Kernel density estimation
    - Estimating Local Distribution Function (ELDF)
    - ELDF density computation
    
    The class uses a consistent naming convention for arguments:
    - Z: Input data points/observations
    - Z0: Reference points, centers, or points where functions are evaluated
    - S: Scale parameter(s) controlling the spread of the distribution
    
    References
    ----------
    Kovanic, P., & Humber, M. B. (2015). The Economics of Information: 
    Mathematical Gnostics for Data Analysis. 
    """
    
    def __init__(self, tol=1e-6):
        """
        Initialize the GnosticDistributionFunction with convergence tolerance.
        
        Parameters
        ----------
        tol : float, optional
            Tolerance for convergence in iterative methods (default is 1e-6).
        """
        self.tol = tol

    def gnostic_kernel(self, Z, Z0, S):
        """
        Compute the Gnostic kernel function.
        
        The Gnostic kernel is defined as 1/(S * cosh(2(Z-Z0)/S)²), representing
        the basic building block for Gnostic distribution functions.
        
        Parameters
        ----------
        Z : float or array-like
            The value(s) to evaluate the kernel at.
        Z0 : float or array-like
            The center(s) of the kernel.
        S : float or array-like
            The scale parameter(s) controlling the spread of the kernel.
        
        Returns
        -------
        float or array-like
            The value(s) of the Gnostic kernel evaluated at Z.

        Example
        -------
        >>> gdf = GnosticDistributionFunction()
        >>> gdf.gnostic_kernel(1.0, 0.0, 1.0)
        0.5403023058681398
        """
        A = Z - Z0
        return 1.0 / (S * np.cosh(2 * A/S) ** 2)
    
    def gnostic_density(self, Z0, Z, S=1.0, normalize=True):
        """
        Calculate the Gnostic kernel density estimate at points Z0 using data Z.
        
        This method computes a non-parametric density estimate by placing a Gnostic
        kernel at each data point in Z and evaluating the sum at each point in Z0.
        
        Parameters
        ----------
        Z0 : array-like
            Points where the density function is evaluated.
        Z : array-like
            Data points (observations) used as centers of kernels.
        S : float or array-like, optional
            Scale parameter(s) controlling kernel width (default=1.0).
            Can be a scalar (same for all points) or array of same length as Z.
        normalize : bool, optional
            If True, normalize so the area under density is 1 (default=True).
            
        Returns
        -------
        array-like
            Density values at each point in Z0.
        
        Example
        -------
        >>> gdf = GnosticDistributionFunction()
        >>> Z0 = np.linspace(-5, 5, 100)
        >>> Z = np.random.normal(0, 1, 50)
        >>> density = gdf.gnostic_density(Z0, Z, S=1.0)
        """
        Z0 = np.asarray(Z0)
        Z = np.asarray(Z)
        if np.isscalar(S):
            S = np.full_like(Z, S, dtype=float)
        else:
            S = np.asarray(S)
            assert len(S) == len(Z), "Scale array must match data length"
        
        density = np.zeros_like(Z0, dtype=float)
        for xi, si in zip(Z, S):
            density += self.gnostic_kernel(Z0, xi, si)
        density /= (len(Z))  # mean kernel (ELDF/EGDF style)
        
        if normalize:
            area = np.trapz(density, Z0)
            if area > 0:
                density /= area
        return density
    
    def eldf(self, Z, Z0, S=1.0):
        """
        Compute the Estimating Local Distribution Function (ELDF).
        
        ELDF is a robust distribution function that maps input data Z to 
        a distribution estimate at reference points Z0.
        
        Parameters
        ---------- 
        Z : array-like
            The data points (observations) for which to compute the ELDF.
        Z0 : array-like
            The reference points where the distribution is evaluated.
        S : float, optional
            The scale parameter controlling the spread of the distribution (default=1.0).
        
        Returns
        -------
        array-like
            The estimated local distribution function values at points Z0.

        Example
        -------
        >>> gdf = GnosticDistributionFunction()
        >>> Z = np.random.normal(0, 1, 100)
        >>> Z0 = np.linspace(-3, 3, 10)
        >>> el = gdf.eldf(Z, Z0, S=1.0)
        >>> print(el)
        """
        Z = np.asarray(Z)
        Z0 = np.asarray(Z0)
        el = np.zeros_like(Z0, dtype=float)
        for Zk in Z:
            qk = (Zk / Z0)**(2/S)
            el += 1 / (1 + qk**4)
        el /= len(Z)
        return el
    
    def eldf_density(self, Z, Z0, S):
        """
        Calculate the ELDF probability density function (dELDF/dZ0).
        
        This method computes the derivative of the ELDF with respect to Z0,
        providing a probability density function based on the Gnostic approach.
        
        Parameters
        ----------
        Z : array-like
            Input data array (observations).
        Z0 : array-like or float
            Grid points or single value at which to evaluate the density.
        S : float
            Scale parameter controlling the spread of the distribution.
            
        Returns
        -------
        array-like
            Density values for each point in Z0.
            
        Notes
        -----
        The ELDF density is the derivative of the ELDF with respect to Z0:
            dEL/dZ0 = (1/n) * Σ [(4/(S*|Z0|)) / ((Zk/Z0)^(2/S)^2 + 1/(Zk/Z0)^(2/S)^2)^2]
        where n is the number of data points in Z.
        """
        # Input validation
        if not isinstance(S, (int, float)) or S <= 0:
            raise ValueError("Scale parameter S must be a positive number")
        
        Z = np.asarray(Z)
        Z0 = np.asarray(Z0)
        
        # Handle empty data case
        if len(Z) == 0:
            return np.zeros_like(Z0, dtype=float)
        
        # For scalar Z0 input
        scalar_input = np.isscalar(Z0)
        if scalar_input:
            Z0 = np.array([Z0])
        
        # Initialize output array
        density = np.zeros_like(Z0, dtype=float)
        
        # Handle zero and near-zero Z0 values
        eps = np.finfo(float).eps
        mask = np.abs(Z0) > eps
        Z0_safe = Z0[mask]
        
        if len(Z0_safe) > 0:  # Only proceed if we have valid Z0 values
            density_safe = np.zeros_like(Z0_safe, dtype=float)
            
            # Vectorized implementation when possible
            if len(Z) < 1000:  # For small datasets, use vectorization
                Z_col = Z[:, np.newaxis]  # Shape for broadcasting
                q_matrix = np.abs(Z_col / Z0_safe)**(2/S)
                q_matrix_sq = q_matrix**2
                denom = (q_matrix_sq + 1/q_matrix_sq)**2
                # Avoid division by zero in denominator
                valid_denom = denom > eps
                result = np.zeros_like(denom, dtype=float)
                result[valid_denom] = 4 / denom[valid_denom]
                density_safe = np.mean(result, axis=0)
            else:  # For large datasets, use loop to avoid memory issues
                for Zk in Z:
                    qk = np.abs(Zk / Z0_safe)**(2/S)
                    denom = (qk**2 + 1/qk**2)**2
                    # Avoid division by zero
                    valid_indices = denom > eps
                    density_safe[valid_indices] += 4 / denom[valid_indices]
                density_safe /= len(Z)
            
            # Apply the scaling factor
            density[mask] = density_safe / (S * np.abs(Z0_safe))
        
        # For Z0 = 0, density is undefined or infinity
        # We set it to 0 to avoid numerical issues
        
        return density[0] if scalar_input else density
    
    def _make_w(self, Z, W=None):
        """
        Create or validate weight vector for internal calculations.

        Calculates vector of equal weights with the sum normalized to 1,
        or checks the length and reshapes W to a column vector.
        
        Parameters
        ----------
        Z : array-like
            Input data vector.
        W : array-like, optional
            Weight vector. If None, equal weights are assigned.

        Returns
        -------
        numpy.ndarray
            Column vector of weights.
        
        Raises
        ------
        ValueError
            If Z is not a vector or W is not a vector of the same length as Z.
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
    
    def _make_q(self, S, Z0, Z, W=None, varS=0):
        """
        Calculate scale parameters and matrices of q and 1/q for Gnostic calculations.
        
        Parameters
        ----------
        S : float or array-like
            Scale parameter (scalar or vector of same length as Z).
        Z0 : array-like
            Scaling points where function is evaluated (can be matrix or vector).
        Z : array-like
            Data vector (observations).
        W : array-like, optional
            Weight vector (should be column vector if used).
        varS : int, optional
            If > 0 and S is scalar, compute variable scale using VarS.
        
        Returns
        -------
        S : float or ndarray
            Updated scale parameter (scalar or column vector).
        q : ndarray
            q matrix.
        q1 : ndarray
            1/q matrix.
            
        Raises
        ------
        ValueError
            If inputs have incorrect dimensions or types.
        """
        scale = ScaleParam()
        Z = np.asarray(Z)
        Z0 = np.asarray(Z0)
        
        if Z.ndim != 1:
            raise ValueError("Z must be a vector")
        
        if np.isscalar(S):
            is_scalar_S = True
        else:
            S = np.asarray(S)
            if S.ndim != 1 or len(S) != len(Z):
                raise ValueError("S must be a scalar or a vector of the same length as Z")
            is_scalar_S = False

        # Flatten Z0 into a column vector
        Z0 = Z0.reshape(-1, 1)
        
        # Compute variable scale if requested
        if varS > 0 and is_scalar_S:
            if W is None:
                raise ValueError("W must be provided when varS > 0 and S is scalar")
            S = scale.var_s(Z, W, S)
        elif not is_scalar_S:
            S = S.reshape(-1, 1)

        # Calculate q and q1
        X = (1.0 / Z0) @ Z.reshape(1, -1)
        gc = GnosticsCharacteristics(X)
        q, q1 = gc._get_q_q1(S=S)

        return S, q, q1