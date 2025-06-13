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
    - Z: Input data points/observations (mandatory)
    - S: Scale parameter(s) controlling the spread of the distribution (mandatory)
    - Z0: Reference points where functions are evaluated (auto-generated if not provided)
    
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
        self.Z = None
        self.S = None
        self.Z0 = None
        self.data_points = 1000  # Default number of points for Z0 generation
        
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
        self.Z = np.asarray(Z)
        
        if np.isscalar(S):
            self.S = np.full_like(Z, S, dtype=float)
        else:
            S = np.asarray(S)
            if len(S) != len(Z):
                raise ValueError("Scale array must match data length")
            self.S = S
            
        # Generate default Z0 points spanning the data range with padding
        padding = 0.1 * (self.Z.max() - self.Z.min())
        self.Z0 = np.linspace(
            max(self.Z.min() - padding, 1e-10),  # Avoid zero or negative values
            self.Z.max() + padding, 
            self.data_points
        )
        
        return self
    
    def set_evaluation_points(self, Z0=None, data_points=None):
        """
        Set custom evaluation points or number of auto-generated points.
        
        Parameters
        ----------
        Z0 : array-like, optional
            Custom evaluation points. If None, points will be auto-generated.
        data_points : int, optional
            Number of points to generate if Z0 is None. If None, uses default.
            
        Returns
        -------
        self : GnosticDistributionFunction
            Returns self for method chaining.
        """
        if Z0 is not None:
            self.Z0 = np.asarray(Z0)
        elif data_points is not None:
            self.data_points = data_points
            if self.Z is not None:
                padding = 0.1 * (self.Z.max() - self.Z.min())
                self.Z0 = np.linspace(
                    max(self.Z.min() - padding, 1e-10),
                    self.Z.max() + padding,
                    self.data_points
                )
        return self
    
    def gnostic_kernel(self, Z0=None):
        """
        Compute the Gnostic kernel function for each data point using vectorized operations.
        
        The Gnostic kernel is defined as 1/(S * cosh(2(Z-Z0)/S)²), representing
        the basic building block for Gnostic distribution functions.
        
        Parameters
        ----------
        Z0 : array-like, optional
            Points where to evaluate the kernel. If None, uses self.Z0.
            
        Returns
        -------
        kernels : ndarray
            Array of shape (len(Z0), len(Z)) where each column is 
            the kernel for a data point evaluated at all Z0 points.
        """
        if self.Z is None or self.S is None:
            raise ValueError("Must call fit(Z, S) before calculating kernels")
            
        Z0 = self.Z0 if Z0 is None else np.asarray(Z0)
        
        # Reshape for broadcasting
        Z0_col = Z0.reshape(-1, 1)  # Shape: (n_points, 1)
        Z_row = self.Z.reshape(1, -1)  # Shape: (1, n_samples)
        S_row = self.S.reshape(1, -1)  # Shape: (1, n_samples)
        
        # Compute kernels using broadcasting (creates a matrix of shape (n_points, n_samples))
        A = Z0_col - Z_row  # Broadcasting: (n_points, 1) - (1, n_samples) = (n_points, n_samples)
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
            Points where to evaluate the density. If None, uses self.Z0.
        normalize : bool, optional
            If True, normalize so the area under density is 1 (default=True).
            
        Returns
        -------
        density : array-like
            Density values at each point in Z0.
        """
        if self.Z is None or self.S is None:
            raise ValueError("Must call fit(Z, S) before calculating density")
            
        Z0 = self.Z0 if Z0 is None else np.asarray(Z0)
        
        # Compute kernels and take mean along axis 1 (samples)
        kernels = self.gnostic_kernel(Z0)
        density = np.mean(kernels, axis=1)
        
        # Normalize if requested
        if normalize:
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
            The reference points where to evaluate the distribution.
            If None, uses self.Z0.
        
        Returns
        -------
        el : array-like
            The estimated local distribution function values at points Z0.
        """
        if self.Z is None or self.S is None:
            raise ValueError("Must call fit(Z, S) before calculating ELDF")
            
        Z0 = self.Z0 if Z0 is None else np.asarray(Z0)
        
        # Reshape data for broadcasting
        Z = self.Z.reshape(-1, 1)  # Shape: (n_samples, 1)
        Z0 = Z0.reshape(1, -1)     # Shape: (1, n_points)
        
        # Handle division-by-zero for Z0 near zero
        eps = np.finfo(float).eps
        Z0_safe = np.maximum(Z0, eps)
        
        # Check if all S values are the same
        if np.all(self.S == self.S[0]):
            # Single scalar S - simple broadcasting
            S = self.S[0]
            qk = (Z / Z0_safe)**(1/S)
            el = np.mean(1 / (1 + qk**4), axis=0)
        else:
            # Varying S values - more complex broadcasting
            S = self.S.reshape(-1, 1)  # Shape: (n_samples, 1)
            
            # Calculate q-values using broadcasting
            # This creates a matrix of shape (n_samples, n_points)
            qk = np.power(Z / Z0_safe, 1/S)
            
            # Calculate ELDF values and take mean along samples axis
            el = np.mean(1 / (1 + qk**4), axis=0)
        
        return el
    
    def eldf_density(self, Z0=None):
        """
        Calculate the ELDF probability density function (dELDF/dZ0) using matrix operations.
        
        This method computes the derivative of the ELDF with respect to Z0,
        providing a probability density function based on the Gnostic approach.
        
        Parameters
        ----------
        Z0 : array-like, optional
            Grid points at which to evaluate the density. If None, uses self.Z0.
            
        Returns
        -------
        density : array-like
            Density values for each point in Z0.
        """
        if self.Z is None or self.S is None:
            raise ValueError("Must call fit(Z, S) before calculating density")
            
        Z0 = self.Z0 if Z0 is None else np.asarray(Z0)
        
        # Handle empty data case
        if len(self.Z) == 0:
            return np.zeros_like(Z0, dtype=float)
        
        # Initialize output array
        density = np.zeros_like(Z0, dtype=float)
        
        # Handle zero and near-zero Z0 values
        eps = np.finfo(float).eps
        mask = np.abs(Z0) > eps
        Z0_safe = Z0[mask]
        
        if len(Z0_safe) > 0:  # Only proceed if we have valid Z0 values
            # Reshape data for broadcasting
            Z = self.Z.reshape(-1, 1)  # Shape: (n_samples, 1)
            Z0_m = Z0_safe.reshape(1, -1)  # Shape: (1, n_points)
            
            # Check if all S values are the same
            if np.all(self.S == self.S[0]):
                # Vectorized calculation for constant S
                S = self.S[0]
                
                # Calculate q-matrix and squared q-matrix
                q_matrix = np.abs(Z / Z0_m)**(1/S)
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
                q_matrix = np.power(np.abs(Z / Z0_m), 1/S)
                q_sq = q_matrix**2
                inv_q_sq = 1/q_sq
                
                # Calculate denominator
                denom = (q_sq + inv_q_sq)**2
                
                # Calculate contribution per point
                valid_denom = denom > eps
                result = np.zeros_like(denom, dtype=float)
                
                # Broadcasting S for division
                S_expanded = np.broadcast_to(S, denom.shape)
                Z0_expanded = np.broadcast_to(np.abs(Z0_m), denom.shape)
                
                # Calculate where denominator is valid
                result[valid_denom] = 4 / (S_expanded[valid_denom] * denom[valid_denom])
                
                # Average over samples
                density_safe = np.mean(result, axis=0)
            
            density[mask] = density_safe
        
        return density
    
    def compute_all(self, Z0=None, normalize=True):
        """
        Compute all Gnostic distribution functions at once using vectorized operations.
        
        Parameters
        ----------
        Z0 : array-like, optional
            Points where to evaluate the functions. If None, uses self.Z0.
        normalize : bool, optional
            If True, normalize densities (default=True).
            
        Returns
        -------
        dict
            Dictionary containing all computed functions:
            - 'Z0': Evaluation points
            - 'kernels': Matrix of individual kernels
            - 'gnostic_density': Gnostic kernel density
            - 'eldf': ELDF function values
            - 'eldf_density': ELDF density values
        """
        if self.Z is None or self.S is None:
            raise ValueError("Must call fit(Z, S) before computing functions")
            
        Z0 = self.Z0 if Z0 is None else np.asarray(Z0)
        
        # Compute all functions
        kernels = self.gnostic_kernel(Z0)
        gnostic_density = np.mean(kernels, axis=1)
        
        # Normalize if requested
        if normalize and len(gnostic_density) > 1:
            area = np.trapz(gnostic_density, Z0)
            if area > 0:
                gnostic_density /= area
        
        # Compute ELDF and ELDF density
        eldf_values = self.eldf(Z0)
        eldf_density_values = self.eldf_density(Z0)
        
        return {
            'Z0': Z0,
            'kernels': kernels,
            'gnostic_density': gnostic_density,
            'eldf': eldf_values,
            'eldf_density': eldf_density_values
        }
        
    def _make_w(self, W=None):
        """
        Create or validate weight vector for internal calculations.

        Calculates vector of equal weights with the sum normalized to 1,
        or checks the length and reshapes W to a column vector.
        
        Parameters
        ----------
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
        if self.Z is None:
            raise ValueError("Must call fit(Z, S) before creating weights")
            
        if W is not None:
            W = np.asarray(W)
            if W.ndim != 1 or len(W) != len(self.Z):
                raise ValueError("W must be a vector of the same length as Z")
            W = W.reshape(-1, 1)
        else:
            W = np.ones(len(self.Z)).reshape(-1, 1) / len(self.Z)
        
        return W