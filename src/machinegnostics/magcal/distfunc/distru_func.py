import numpy as np
from scipy.optimize import minimize
from machinegnostics.magcal.data_conversion import DataConversion

class DistributionFunctions:
    """
    Base class for Gnostic distribution functions.
    
    Gnostic distribution functions are a family of robust statistical methods based on 
    the principles of information theory. This implementation provides improved integration
    between EDF, ELDF and density calculations.
    """
    
    def __init__(self, tol=1e-6, data_form=None, data_points=1000, data_lb=None, data_ub=None):
        """
        Initialize the DistributionFunction.
        
        Parameters
        ----------
        tol : float, optional
            Tolerance for convergence in iterative methods (default is 1e-6).
        data_form : str, optional
            Data form: 'a' for additive, 'm' for multiplicative (default is None).
        data_points : int, optional
            Default number of evaluation points to generate (default is 1000).
        data_lb : float, optional
            Lower bound for data. If None, will be set from data.
        data_ub : float, optional
            Upper bound for data. If None, will be set from data.
        """
        self.tol = tol
        self.Z = None    # Original data
        self.Zi = None   # Data in infinite domain
        self.S = None    # Scale parameter
        self.Z0 = None   # Evaluation points in original domain
        self.Z0i = None  # Evaluation points in infinite domain
        self.data_points = data_points
        self.data_form = data_form  # 'a' for additive or 'm' for multiplicative
        self.data_lb = data_lb  # Lower bound for data
        self.data_ub = data_ub  # Upper bound for data

        # Data bounds attributes
        self.fin_lb = None
        self.fin_ub = None
        
    def fit(self, Z, S):
        """
        Fit the distribution with data and scale parameter.
        
        Parameters
        ----------
        Z : array-like
            Data points (observations).
        S : float or array-like
            Scale parameter(s) controlling the spread of the distribution.
            
        Returns
        -------
        self : DistributionFunction
            Returns self for method chaining.
        """
        # Store original data
        self.Z = np.asarray(Z)
        
        # Validate input data
        if self.Z.ndim != 1:
            raise ValueError("Z must be a 1D array of data points")
        if len(self.Z) == 0:
            raise ValueError("Z must contain at least one data point")
            
        # Process scale parameter
        if np.isscalar(S):
            self.S = np.full_like(self.Z, S, dtype=float)
        else:
            S = np.asarray(S)
            if len(S) != len(self.Z):
                raise ValueError("Scale array must match data length")
            self.S = S
            
        # Set data bounds from original data
        if self.data_lb is None:
            self.data_lb = self.Z.min()
        if self.data_ub is None:
            self.data_ub = self.Z.max()
        
        # Generate default Z0 points in original domain
        padding = 0.1 * (self.Z.max() - self.Z.min())
        self.Z0 = np.linspace(
            max(self.Z.min() - padding, 1e-10),  # Avoid zero or negative values
            self.Z.max() + padding, 
            self.data_points
        )
        
        # Transform data to working domain if needed
        if self.data_form == 'm' or self.data_form == 'a':
            # Convert to working domain
            self.Zi = self._to_working_domain(self.Z)
            self.Z0i = self._to_working_domain(self.Z0)
        elif self.data_form is None:
            self.Zi = self.Z
            self.Z0i = self.Z0
        else:
            raise ValueError("data_form must be 'a' for additive or 'm' for multiplicative")
        return self
    
    def set_evaluation_points(self, Z0=None, data_points=None):
        """
        Set custom evaluation points or number of auto-generated points.
        
        Parameters
        ----------
        Z0 : array-like, optional
            Custom evaluation points in original domain.
        data_points : int, optional
            Number of points to generate if Z0 is None.
            
        Returns
        -------
        self : DistributionFunction
            Returns self for method chaining.
        """
        if Z0 is not None:
            # Use provided evaluation points
            self.Z0 = np.asarray(Z0)
            # Transform to working domain 
            if self.data_form == 'm' or self.data_form == 'a':
                self.Z0i = self._to_working_domain(self.Z0)
            elif self.data_form is None:
                self.Z0i = self.Z0
            else:
                raise ValueError("data_form must be 'a' for additive or 'm' for multiplicative")
        elif data_points is not None:
            # Update data points count and regenerate points
            self.data_points = data_points
            if self.Z is not None:
                padding = 0.1 * (self.Z.max() - self.Z.min())
                self.Z0 = np.linspace(
                    max(self.Z.min() - padding, 1e-10),
                    self.Z.max() + padding,
                    self.data_points
                )
                # Transform to working domain if needed
                if self.data_form == 'm' or self.data_form == 'a':
                    self.Z0i = self._to_working_domain(self.Z0)
                elif self.data_form is None:
                    self.Z0i = self.Z0
                else:
                    raise ValueError("data_form must be 'a' for additive or 'm' for multiplicative")
        return self
    
    def _to_working_domain(self, Z):
        """
        Transform data to the working domain based on data form.
        
        For additive data, this transforms to a suitable working domain.
        For multiplicative data, this handles negative values and transforms appropriately.
        
        Parameters
        ----------
        Z : array-like
            Data in original domain
            
        Returns
        -------
        Z_working : array-like
            Data in working domain
        """
        if Z is None:
            return None
            
        Z = np.asarray(Z)
        dc = DataConversion()
        
        if self.data_form == 'm':
            # For multiplicative data, handle negative values and transform to appropriate domain
            Z_positive = dc._convert_mz(Z, Z.min(), Z.max())
            # finite domain bounds
            self.fin_lb = Z_positive.min()
            self.fin_ub = Z_positive.max()
            # Transform from finite to infinite domain if needed
            Zi = dc._convert_fininf(Z_positive, self.fin_lb, self.fin_ub)
            return Zi
        elif self.data_form == 'a':
            # For additive data, handle as needed
            Z_positive = dc._convert_az(Z, Z.min(), Z.max())
            # Store finite domain bounds
            self.fin_lb = Z_positive.min()
            self.fin_ub = Z_positive.max()
            # Transform from finite to infinite domain if needed
            Zi = dc._convert_fininf(Z_positive, self.fin_lb, self.fin_ub)
            return Zi
        elif self.data_form is None:
            # If no specific data form, just return the original data
            return Z
        else:
            raise ValueError("data_form must be 'a' for additive or 'm' for multiplicative")

    def _from_working_domain(self, Z_working):
        """
        Transform data from working domain back to original domain.
        
        Parameters
        ----------
        Z_working : array-like
            Data in working domain
            
        Returns
        -------
        Z : array-like
            Data in original domain
        """
        if Z_working is None:
            return None
            
        Z_working = np.asarray(Z_working)
        dc = DataConversion()
        
        if self.data_form == 'm':
            # For multiplicative data, reverse the transformations
            # First convert from infinite to finite domain
            Z_finite = dc._convert_inffin(Z_working, self.fin_lb, self.fin_ub)
            # Then reverse the mz conversion
            Z_original = dc._convert_zm(Z_finite, self.data_lb, self.data_ub)
            return Z_original
        elif self.data_form == 'a':
            # For additive data, reverse the transformations
            # First convert from infinite to finite domain
            Z_finite = dc._convert_inffin(Z_working, self.fin_lb, self.fin_ub)
            # Then reverse the az conversion
            Z_original = dc._convert_za(Z_finite, self.data_lb, self.data_ub)
            return Z_original
        elif self.data_form is None:
            # If no specific data form, just return the original data
            return Z_working
        else:
            raise ValueError("data_form must be 'a' for additive or 'm' for multiplicative")
    
    def generate_ks_points(self, num_points=None):
        """
        Generate Kolmogorov-Smirnov points for distribution fitting.
        
        Parameters
        ----------
        num_points : int, optional
            Number of points to generate. Defaults to data length.
            
        Returns
        -------
        Z0 : ndarray
            Generated points in the original domain.
        ks_probs : ndarray
            Corresponding probabilities.
        """
        if self.Z is None:
            raise ValueError("Must call fit(Z, S) before generating K-S points")
        
        # Use data length if not specified
        L = num_points if num_points is not None else len(self.Z)
        
        # Generate K-S probabilities
        ks_probs = np.arange(1, 2*L, 2) / (2*L)
        
        # Generate corresponding points
        data_range = self.data_ub - self.data_lb
        Z0 = self.data_lb + data_range * ks_probs
        
        return Z0, ks_probs
    
    def gnostic_kernel(self, Z0=None):
        """
        Compute the Gnostic kernel function for each data point.
        
        Parameters
        ----------
        Z0 : array-like, optional
            Points where to evaluate the kernel in original domain. 
            
        Returns
        -------
        kernels : ndarray
            Array of shape (len(Z0), len(Z)) where each column is 
            the kernel for a data point evaluated at all Z0 points.
        """
        if self.Z is None or self.S is None:
            raise ValueError("Must call fit(Z, S) before calculating kernels")
            
        # Use default evaluation points if not provided
        if Z0 is None:
            Z0 = self.Z0
            Z0_working = self.Z0i
        else:
            Z0 = np.asarray(Z0)
            # Transform to working domain if needed
            if self.data_form == 'm' or self.data_form == 'a':
                Z0_working = self._to_working_domain(Z0)
            elif self.data_form is None:
                Z0_working = Z0
            else:
                raise ValueError("data_form must be 'a' for additive or 'm' for multiplicative")
        
        # Work in appropriate domain based on data form
        Z0_col = Z0_working.reshape(-1, 1)  # Shape: (n_points, 1)
        Z_row = self.Zi.reshape(1, -1)      # Shape: (1, n_samples)
        S_row = self.S.reshape(1, -1)       # Shape: (1, n_samples)
        
        # Compute kernels using broadcasting
        if self.data_form == 'm':
            # For multiplicative data, use ratio
            A = Z0_col / Z_row
        elif self.data_form == 'a' or self.data_form is None:
            # For additive data, use difference
            A = Z0_col - Z_row

        cosh_term = np.cosh(2 * A / S_row)
        kernels = 1.0 / (S_row * cosh_term ** 2)
        
        return kernels
    
    def gnostic_density(self, Z0=None, normalize=True):
        """
        Calculate the Gnostic kernel density estimate.
        
        Parameters
        ----------
        Z0 : array-like, optional
            Points where to evaluate the density in original domain.
        normalize : bool, optional
            If True, normalize so the area under density is 1.
            
        Returns
        -------
        density : array-like
            Density values at each point in Z0 in original domain.
        """
        if self.Z is None or self.S is None:
            raise ValueError("Must call fit(Z, S) before calculating density")
            
        # Use default evaluation points if not provided
        if Z0 is None:
            Z0 = self.Z0
        else:
            Z0 = np.asarray(Z0)
        
        # Compute kernels and take mean along axis 1 (samples)
        kernels = self.gnostic_kernel(Z0)
        density = np.mean(kernels, axis=1)
        
        # Normalize if requested
        if normalize and len(density) > 1:
            # For multiplicative data, adjust for log-transformation
            if self.data_form == 'm':
                # Integration in log domain needs adjustment
                area = np.trapz(density * Z0, np.log(Z0))
            else:
                area = np.trapz(density, Z0)
                
            if area > 0:
                density /= area
                
        return density
    
    def edf(self, Z0=None, smoothed=False, num_points=100, density_func=None):
        """
        Compute the Empirical Distribution Function (EDF) using various density functions.
        
        Parameters
        ---------- 
        Z0 : array-like, optional
            The reference points where to evaluate the empirical distribution.
        smoothed : bool, optional
            If True, returns a smoothed version of the EDF using kernels.
        num_points : int, optional
            Number of points for smoothed EDF calculation.
        density_func : callable, optional
            Function to use for density calculation. Should take Z0 as input and return density values.
            If None, uses gnostic_kernel for smoothed EDF or step function for non-smoothed.
            
        Returns
        -------
        Z0 : array-like
            Evaluation points.
        edf_values : array-like
            The empirical distribution function values.
        """
        if self.Z is None:
            raise ValueError("Must call fit(Z, S) before calculating EDF")
            
        # Use K-S points if Z0 is not provided
        if Z0 is None:
            Z0, _ = self.generate_ks_points(num_points)
        else:
            Z0 = np.asarray(Z0)
        
        # Sort Z0 to ensure proper integration
        sort_indices = np.argsort(Z0)
        Z0_sorted = Z0[sort_indices]
        
        if not smoothed and density_func is None:
            # Standard step-function EDF
            sorted_data = np.sort(self.Z)
            edf_values = np.zeros_like(Z0_sorted, dtype=float)
            
            for i, z in enumerate(Z0_sorted):
                # Count data points less than or equal to z
                edf_values[i] = np.sum(sorted_data <= z) / len(sorted_data)
        else:
            # Use provided density function or default to gnostic kernel
            if density_func is None:
                # Transform to working domain if needed
                if self.data_form in ('a', 'm'):
                    # Compute kernels for each evaluation point
                    kernels = self.gnostic_kernel(Z0_sorted)
                    # Calculate density
                    density = np.mean(kernels, axis=1)
                else:
                    # Compute kernels for each evaluation point
                    kernels = self.gnostic_kernel(Z0_sorted)
                    # Calculate density
                    density = np.mean(kernels, axis=1)
            else:
                # Use provided density function
                density = density_func(Z0_sorted)
            
            # Ensure density is non-negative
            density = np.maximum(density, 0)
            
            # Compute CDF by numerical integration
            # For the first point, CDF is 0
            edf_values = np.zeros_like(density)
            
            # Calculate areas between consecutive points
            if len(Z0_sorted) > 1:
                # Calculate width of each bin
                widths = np.diff(Z0_sorted)
                
                # Calculate area using trapezoidal rule
                areas = 0.5 * widths * (density[:-1] + density[1:])
                
                # Accumulate areas for CDF
                edf_values[1:] = np.cumsum(areas)
                
                # Normalize to [0, 1]
                if edf_values[-1] > 0:
                    edf_values /= edf_values[-1]
        
        # Reorder to match original Z0 if it wasn't sorted
        if not np.array_equal(Z0, Z0_sorted):
            inv_sort_indices = np.argsort(sort_indices)
            edf_values = edf_values[inv_sort_indices]
        
        return Z0, edf_values
    
    def _calculate_eldf_values(self, Z0_working):
        """
        Helper method to calculate ELDF values in the working domain.
        
        Parameters
        ----------
        Z0_working : array-like
            Evaluation points in working domain.
        
        Returns
        -------
        eldf_values : array-like
            ELDF values at the evaluation points.
        """
        # Reshape for broadcasting
        Z_working = self.Zi.reshape(-1, 1)     # Shape: (n_samples, 1)
        Z0_w = Z0_working.reshape(1, -1)       # Shape: (1, n_points)
        
        # Handle division-by-zero
        eps = np.finfo(float).eps
        Z0_safe = np.maximum(Z0_w, eps)
        
        # Check if all S values are the same
        if np.all(self.S == self.S[0]):
            # Single scalar S - simple broadcasting
            S = self.S[0]
            # Calculate ratio in working domain
            qk = (Z_working / Z0_safe) ** (1 / S)
            eldf_values = np.mean(1 / (1 + qk**4), axis=0)
        else:
            # Varying S values - more complex broadcasting
            S = self.S.reshape(-1, 1)  # Shape: (n_samples, 1)
            
            # Calculate ratio in working domain
            qk = (Z_working / Z0_safe) ** (1 / S)
            
            # Calculate ELDF values and take mean along samples axis
            eldf_values = np.mean(1 / (1 + qk**4), axis=0)
        
        return eldf_values
    
    def eldf(self, Z0=None, use_edf=False, num_points=100):
        """
        Compute the Estimating Local Distribution Function (ELDF) using K-S points.
        
        Parameters
        ---------- 
        Z0 : array-like, optional
            The reference points where to evaluate the distribution.
            If None, K-S points will be used.
        use_edf : bool, optional
            If True, uses the smoothed EDF approach for improved robustness.
        num_points : int, optional
            Number of K-S points to generate if Z0 is None.
        
        Returns
        -------
        Z0 : array-like
            Evaluation points in original domain.
        eldf_values : array-like
            The estimated local distribution function values.
        """
        if self.Z is None or self.S is None:
            raise ValueError("Must call fit(Z, S) before calculating ELDF")
            
        # Generate K-S points if Z0 is not provided
        if Z0 is None:
            Z0, _ = self.generate_ks_points(num_points)
        else:
            Z0 = np.asarray(Z0)
        
        if use_edf:
            # Create a density function for ELDF calculation
            def eldf_density_func(z):
                # Transform to working domain
                if self.data_form in ('a', 'm'):
                    z_working = self._to_working_domain(z)
                else:
                    z_working = z
                    
                # Calculate ELDF density values in working domain
                values = self._calculate_eldf_values(z_working)
                
                return values
            
            # Use the EDF function with our custom density function
            _, eldf_values = self.edf(Z0, smoothed=True, density_func=eldf_density_func)
            return Z0, eldf_values
        
        # For traditional calculation, transform to working domain
        if self.data_form in ('a', 'm'):
            Z0_working = self._to_working_domain(Z0)
        else:
            Z0_working = Z0
        
        # Calculate ELDF values in working domain
        eldf_values = self._calculate_eldf_values(Z0_working)
        
        return Z0, eldf_values
    
    def eldf_density(self, Z0=None, use_kernels=False, num_points=100):
        """
        Calculate the ELDF probability density function using K-S points.
        
        Parameters
        ----------
        Z0 : array-like, optional
            Grid points at which to evaluate the density.
            If None, K-S points will be used.
        use_kernels : bool, optional
            If True, uses direct kernel estimation for improved robustness.
        num_points : int, optional
            Number of K-S points to generate if Z0 is None.
            
        Returns
        -------
        Z0 : array-like
            Evaluation points in original domain.
        density : array-like
            Density values for each point in Z0.
        """
        if self.Z is None or self.S is None:
            raise ValueError("Must call fit(Z, S) before calculating density")
            
        # Generate K-S points if Z0 is not provided
        if Z0 is None:
            Z0, _ = self.generate_ks_points(num_points)
        else:
            Z0 = np.asarray(Z0)
        
        if use_kernels:
            # First transform to working domain if needed
            if self.data_form in ('a', 'm'):
                Z0_working = self._to_working_domain(Z0)
            else:
                Z0_working = Z0
                
            # Get smoothed ELDF values
            _, eldf_values = self.eldf(Z0, use_edf=True, num_points=num_points)
            
            # To get density, approximate derivative of ELDF
            density = np.zeros_like(eldf_values)
            
            # Use finite differences to approximate derivative
            if len(Z0) > 2:
                # Ensure Z0 is sorted for differentiation
                sort_indices = np.argsort(Z0)
                Z0_sorted = Z0[sort_indices]
                eldf_sorted = eldf_values[sort_indices]
                
                # For interior points, use central differences
                density_sorted = np.zeros_like(eldf_sorted)
                
                # First point: forward difference
                density_sorted[0] = (eldf_sorted[1] - eldf_sorted[0]) / (Z0_sorted[1] - Z0_sorted[0])
                
                # Interior points: central differences
                for i in range(1, len(Z0_sorted)-1):
                    density_sorted[i] = (eldf_sorted[i+1] - eldf_sorted[i-1]) / (Z0_sorted[i+1] - Z0_sorted[i-1])
                
                # Last point: backward difference
                density_sorted[-1] = (eldf_sorted[-1] - eldf_sorted[-2]) / (Z0_sorted[-1] - Z0_sorted[-2])
                
                # Ensure non-negative density
                density_sorted = np.maximum(density_sorted, 0)
                
                # Restore original order
                inv_sort_indices = np.argsort(sort_indices)
                density = density_sorted[inv_sort_indices]
                
                # Normalize if there are positive values
                if np.sum(density) > 0:
                    # For multiplicative data, use log-space integration
                    if self.data_form == 'm':
                        area = np.trapz(density * Z0, np.log(Z0))
                    else:
                        area = np.trapz(density, Z0)
                    
                    if area > 0:
                        density /= area
        
            return Z0, density
        
        # Traditional calculation with domain transformations
        
        # Transform to working domain
        if self.data_form in ('a', 'm'):
            Z0_working = self._to_working_domain(Z0)
        else:
            Z0_working = Z0
        
        # Handle empty data case
        if len(self.Zi) == 0:
            return Z0, np.zeros_like(Z0, dtype=float)
        
        # Initialize output array
        density = np.zeros_like(Z0_working, dtype=float)
        
        # Handle zero and near-zero Z0 values
        eps = np.finfo(float).eps
        mask = np.abs(Z0_working) > eps
        Z0_safe = Z0_working[mask]
        
        if len(Z0_safe) > 0:  # Only proceed if we have valid Z0 values
            # Reshape data for broadcasting
            Z_working = self.Zi.reshape(-1, 1)   # Shape: (n_samples, 1)
            Z0_w = Z0_safe.reshape(1, -1)        # Shape: (1, n_points)
            
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
                
                # Average over samples and scale by S
                density_safe = np.mean(result, axis=0) / S
                result_mean = density_safe
                    
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
                
                # Average over samples to get a 1D result
                result_mean = np.mean(result, axis=0)
                            
            density[mask] = result_mean
        
        return Z0, density
    
    def optimize_parameters(self, num_points=100, bounds=None):
        """
        Optimize distribution parameters using Kolmogorov-Smirnov criterion.
        
        Parameters
        ----------
        num_points : int, optional
            Number of points for K-S estimation.
        bounds : tuple of tuples, optional
            Bounds for (lower_bound, upper_bound, scale) parameters.
            
        Returns
        -------
        dict
            Dictionary with optimized parameters.
        """
        if self.Z is None:
            raise ValueError("Must set data with fit() before optimizing parameters")
        
        # Default bounds if not provided
        if bounds is None:
            data_min, data_max = self.Z.min(), self.Z.max()
            data_range = data_max - data_min
            bounds = [
                (data_min - 0.5*data_range, data_min + 0.1*data_range),  # LB
                (data_max - 0.1*data_range, data_max + 0.5*data_range),  # UB
                (0.001, 2.0)  # S
            ]
        
        # Initial parameter values
        initial_params = [
            self.data_lb if self.data_lb is not None else self.Z.min(),
            self.data_ub if self.data_ub is not None else self.Z.max(),
            np.mean(self.S) if self.S is not None else 1.0
        ]
        
        # Define criterion function for optimization
        def criterion_function(params):
            lb, ub, s_scalar = params
            if lb >= ub or s_scalar <= 0:
                return np.inf  # Invalid configuration
            
            # Generate K-S points
            Z0 = np.linspace(lb, ub, num_points)
            ks_probs = np.arange(1, 2*num_points, 2) / (2*num_points)
            
            # Temporarily set parameters
            orig_S = self.S.copy() if self.S is not None else None
            S_temp = np.ones_like(self.Z) * s_scalar
            self.S = S_temp
            
            # Calculate smoothed EDF
            _, smooth_cdf = self.edf(Z0, smoothed=True)
            
            # Calculate difference from theoretical K-S probabilities
            diffs = np.abs(smooth_cdf - ks_probs)
            cf = np.mean(diffs)  # Could use np.max for strict K-S test
            
            # Restore original S
            self.S = orig_S
            
            return cf
        
        # Run optimization
        result = minimize(criterion_function, initial_params, bounds=bounds, method='L-BFGS-B')
        
        # Extract optimized parameters
        opt_lb, opt_ub, opt_s = result.x
        
        return {
            'data_lb': opt_lb,
            'data_ub': opt_ub,
            'scale': opt_s,
            'success': result.success,
            'message': result.message,
            'criterion_value': result.fun
        }
    
    def get_functions(self, Z0=None, normalize=True, use_smoothed_edf=True):
        """
        Get all distribution functions at specified evaluation points.
        
        Parameters
        ----------
        Z0 : array-like, optional
            Points where to evaluate the functions.
            If None, K-S points will be used.
        normalize : bool, optional
            If True, normalize densities (default=True).
        use_smoothed_edf : bool, optional
            If True, use smoothed EDF for better integration.
            
        Returns
        -------
        dict
            Dictionary containing all computed functions.
        """
        if self.Z is None or self.S is None:
            raise ValueError("Must call fit(Z, S) before computing functions")
            
        # Use default evaluation points if not provided
        if Z0 is None:
            Z0, _ = self.generate_ks_points()
        else:
            Z0 = np.asarray(Z0)
            
        # Compute functions
        kernels = self.gnostic_kernel(Z0)
        gnostic_density = self.gnostic_density(Z0, normalize=normalize)
        Z0, edf_values = self.edf(Z0, smoothed=use_smoothed_edf)
        Z0, eldf_values = self.eldf(Z0, use_edf=use_smoothed_edf)
        Z0, eldf_density_values = self.eldf_density(Z0, use_kernels=use_smoothed_edf)

        return {
            'Z0': Z0,                         # Original domain evaluation points
            'kernels': kernels,               # Kernels at each evaluation point
            'gnostic_density': gnostic_density,    # Gnostic density
            'eldf': eldf_values,              # ELDF values
            'eldf_density': eldf_density_values,   # ELDF density
            'edf': edf_values                 # EDF values
        }
    
    def get_ks_points(self):
        """
        Get the Kolmogorov-Smirnov (K-S) points used for distribution fitting.
        
        Returns
        -------
        dict
            Dictionary containing K-S points information.
        """
        if self.Z is None:
            raise ValueError("Must call fit(Z, S) before calculating K-S points")
        
        # Sample size
        L = len(self.Z)
        
        # K-S probability points: 1/(2L), 3/(2L), ..., (2L-1)/(2L)
        ks_probs = np.arange(1, 2*L, 2) / (2*L)
        
        # K-S quantile points: sorted data
        sorted_data = np.sort(self.Z)
        
        return {
            'probabilities': ks_probs,
            'quantiles': sorted_data
        }