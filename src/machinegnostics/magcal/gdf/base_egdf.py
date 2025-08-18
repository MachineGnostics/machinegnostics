"""
base class for EGDF
EGDF - Estimating Global Distribution Function.

Author: Nirmal Parmar
Machine Gnostics
"""

import numpy as np
import warnings
from scipy.optimize import minimize
from machinegnostics.magcal.characteristics import GnosticsCharacteristics
from machinegnostics.magcal.gdf.base_df import BaseDistFunc
from machinegnostics.magcal.data_conversion import DataConversion
from machinegnostics.magcal.gdf.wedf import WEDF
from machinegnostics.magcal.mg_weights import GnosticsWeights


class BaseEGDF(BaseDistFunc):
    """
    Base class for EGDF (Estimating Global Distribution Function).
    
    This class provides a comprehensive framework for estimating global distribution
    functions with optimization capabilities and derivative analysis.
    """
    
    # Class constants for optimization bounds
    _OPTIMIZATION_BOUNDS = {
        'S_MIN': 0.05, 'S_MAX': 100.0,
        'LB_MIN': 1e-6, 'LB_MAX': np.exp(-1.000001),
        'UB_MIN': np.exp(1.000001), 'UB_MAX': 1e6,
        'Z0_SEARCH_FACTOR': 0.1  # For Z0 search range
    }
    
    # Numerical constants
    _NUMERICAL_EPS = np.finfo(float).eps
    _NUMERICAL_MAX = 1e6
    _DERIVATIVE_TOLERANCE = 1e-6
    # _Z0_OPTIMIZATION_TOLERANCE = 1e-8

    def __init__(self,
                 data: np.ndarray,
                 DLB: float = None,
                 DUB: float = None,
                 LB: float = None,
                 UB: float = None,
                 S = 'auto',
                 tolerance: float = 1e-3,
                 data_form: str = 'a',
                 n_points: int = 500,
                 homogeneous: bool = True,
                 catch: bool = True,
                 weights: np.ndarray = None,
                 wedf: bool = True,
                 opt_method: str = 'L-BFGS-B',
                 verbose: bool = False,
                 max_data_size: int = 1000,
                 flush: bool = True):
        """Initialize the EGDF class with comprehensive validation."""
        
        # Store raw inputs
        self.data = data
        self.DLB = DLB
        self.DUB = DUB
        self.LB = LB
        self.UB = UB
        self.S = S
        self.tolerance = tolerance
        self.data_form = data_form
        self.n_points = n_points
        self.homogeneous = homogeneous
        self.catch = catch
        self.weights = weights if weights is not None else np.ones_like(data)
        self.wedf = wedf
        self.opt_method = opt_method
        self.verbose = verbose
        self.max_data_size = max_data_size
        self.flush = flush
        
        # Initialize state variables
        self.params = {}
        self._fitted = False
        self._derivatives_calculated = False
        self._marginal_analysis_done = False
        
        # Initialize computation cache
        self._computation_cache = {
            'data_converter': None,
            'characteristics_computer': None,
            'weights_normalized': None,
            'smooth_curves_generated': False
        }
        
        # Validate all inputs
        self._validate_inputs()
        
        # Store initial parameters if catching
        if self.catch:
            self._store_initial_params()

    # =============================================================================
    # VALIDATION AND INITIALIZATION
    # =============================================================================
    
    def _validate_inputs(self):
        """Comprehensive input validation."""
        # Data validation
        if not isinstance(self.data, np.ndarray):
            raise ValueError("Data must be a numpy array.")
        if self.data.size == 0:
            raise ValueError("Data array cannot be empty.")
        if not np.isfinite(self.data).all():
            raise ValueError("Data must contain only finite values.")
        
        # Bounds validation
        for bound, name in [(self.DLB, 'DLB'), (self.DUB, 'DUB'), (self.LB, 'LB'), (self.UB, 'UB')]:
            if bound is not None and (not isinstance(bound, (int, float)) or not np.isfinite(bound)):
                raise ValueError(f"{name} must be a finite numeric value or None.")
        
        # Parameter validation
        if not isinstance(self.S, (int, float, str)):
            raise ValueError("S must be a numeric positive value or 'auto'.")
        if isinstance(self.S, (int, float)) and self.S <= 0:
            raise ValueError("S must be positive when specified as a number.")
        
        if not isinstance(self.tolerance, (int, float)) or self.tolerance <= 0:
            raise ValueError("Tolerance must be a positive numeric value.")
        
        if self.data_form not in ['a', 'm']:
            raise ValueError("data_form must be 'a' for additive or 'm' for multiplicative.")
        
        if not isinstance(self.n_points, int) or self.n_points <= 0:
            raise ValueError("n_points must be a positive integer.")
        
        # Weights validation
        if self.weights is not None:
            if not isinstance(self.weights, np.ndarray):
                raise ValueError("weights must be a numpy array.")
            if len(self.weights) != len(self.data):
                raise ValueError("Weights must have the same length as data.")
            if not np.all(self.weights >= 0):
                raise ValueError("All weights must be non-negative.")
        
        # flush parameter validation
        if not isinstance(self.flush, bool):
            raise ValueError("flush must be a boolean value.")
        # if length of data exceeds max_data_size, set flush to True
        if len(self.data) > self.max_data_size and not self.flush:
            warnings.warn(f"Data size ({len(self.data)}) exceeds max_data_size ({self.max_data_size}). "
                          "For optimal compute performance, set 'flush=True' or increase 'max_data_size'.")
            self.flush = True

        # Boolean parameters
        for param, name in [(self.homogeneous, 'homogeneous'), (self.catch, 'catch'), 
                           (self.wedf, 'wedf'), (self.verbose, 'verbose')]:
            if not isinstance(param, bool):
                raise ValueError(f"{name} must be a boolean value.")

    def _store_initial_params(self):
        """Store initial parameters for reference."""
        self.params.update({
            'data': np.sort(self.data).copy(),
            'DLB': self.DLB,
            'DUB': self.DUB,
            'LB': self.LB,
            'UB': self.UB,
            'S': self.S,
            'tolerance': self.tolerance,
            'data_form': self.data_form,
            'n_points': self.n_points,
            'homogeneous': self.homogeneous,
            'weights': self.weights.copy() if self.weights is not None else None
        })

    # =============================================================================
    # DATA PREPROCESSING AND TRANSFORMATION
    # =============================================================================
    
    def _get_data_converter(self):
        """Get or create cached data converter."""
        if self._computation_cache['data_converter'] is None:
            self._computation_cache['data_converter'] = DataConversion()
        return self._computation_cache['data_converter']

    def _estimate_data_bounds(self):
        """Estimate data bounds (DLB and DUB) if not provided."""
        if self.DLB is None:
            self.DLB = np.min(self.data)
        if self.DUB is None:
            self.DUB = np.max(self.data)
        
        # Validate bounds
        if self.DLB >= self.DUB:
            raise ValueError("DLB must be less than DUB.")
        
        if self.catch:
            self.params.update({'DLB': float(self.DLB), 'DUB': float(self.DUB)})

    def _estimate_weights(self):
        """Process and normalize weights."""
        if self.weights is None:
            self.weights = np.ones_like(self.data, dtype=float)
        else:
            self.weights = np.asarray(self.weights, dtype=float)
        
        # Normalize weights to sum to n (number of data points)
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights = self.weights / weight_sum * len(self.weights)
        else:
            raise ValueError("Sum of weights must be positive.")
        
        # Apply gnostic weights for non-homogeneous data
        if not self.homogeneous:
            gw = GnosticsWeights()
            self.gweights = gw._get_gnostic_weights(self.z)
            self.weights = self.gweights * self.weights
        
        # Cache normalized weights
        self._computation_cache['weights_normalized'] = self.weights.copy()
        
        if self.catch:
            self.params['weights'] = self.weights.copy()

    def _transform_data_to_standard_domain(self):
        """Transform data to standard z-domain."""
        dc = self._get_data_converter()
        
        if self.data_form == 'a':
            self.z = dc._convert_az(self.data, self.DLB, self.DUB)
        elif self.data_form == 'm':
            self.z = dc._convert_mz(self.data, self.DLB, self.DUB)
        
        if self.catch:
            self.params['z'] = self.z.copy()

    def _generate_evaluation_points(self):
        """Generate points for smooth evaluation."""
        self.di_points_n = np.linspace(self.DLB, self.DUB, self.n_points)

        dc = self._get_data_converter()
        if self.data_form == 'a':
            self.z_points_n = dc._convert_az(self.di_points_n, self.DLB, self.DUB)
        else:
            self.z_points_n = dc._convert_mz(self.di_points_n, self.DLB, self.DUB)
        
        if self.catch:
            self.params.update({
                'z_points': self.z_points_n.copy(),
                'di_points': self.di_points_n.copy()
            })

    # =============================================================================
    # BOUNDS ESTIMATION
    # =============================================================================
    
    def _estimate_initial_probable_bounds(self):
        """Estimate initial probable bounds (LB and UB)."""
        dc = self._get_data_converter()
        
        # Estimate LB if not provided
        if self.LB is None:
            if self.data_form == 'a':
                pad = (self.DUB - self.DLB) / 2
                lb_raw = self.DLB - pad
                self.LB_init = dc._convert_az(lb_raw, self.DLB, self.DUB)
            elif self.data_form == 'm':
                lb_raw = self.DLB / np.sqrt(self.DUB / self.DLB)
                self.LB_init = dc._convert_mz(lb_raw, self.DLB, self.DUB)
        else:
            if self.data_form == 'a':
                self.LB_init = dc._convert_az(self.LB, self.DLB, self.DUB)
            else:
                self.LB_init = dc._convert_mz(self.LB, self.DLB, self.DUB)

        # Estimate UB if not provided
        if self.UB is None:
            if self.data_form == 'a':
                pad = (self.DUB - self.DLB) / 2
                ub_raw = self.DUB + pad
                self.UB_init = dc._convert_az(ub_raw, self.DLB, self.DUB)
            elif self.data_form == 'm':
                ub_raw = self.DUB * np.sqrt(self.DUB / self.DLB)
                self.UB_init = dc._convert_mz(ub_raw, self.DLB, self.DUB)
        else:
            if self.data_form == 'a':
                self.UB_init = dc._convert_az(self.UB, self.DLB, self.DUB)
            else:
                self.UB_init = dc._convert_mz(self.UB, self.DLB, self.DUB)

        if self.catch:
            self.params.update({'LB_init': self.LB_init, 'UB_init': self.UB_init})

    # =============================================================================
    # DISTRIBUTION FUNCTION COMPUTATION
    # =============================================================================
    
    def _get_distribution_function_values(self, use_wedf=True):
        """Get WEDF or KS points for optimization."""
        if use_wedf:
            wedf_ = WEDF(self.data, weights=self.weights, data_lb=self.DLB, data_ub=self.DUB)
            # if smooth:
            #     df_values = wedf_.fit(self.di_points_n)
            # else:
            df_values = wedf_.fit(self.data)
            
            if self.catch:
                self.params['wedf'] = df_values.copy()
            
            if self.verbose:
                print("WEDF values computed.")
            return df_values
        else:
            # n_points = self.n_points if smooth else len(self.data)
            df_values = self._generate_ks_points(len(self.data))
            
            if self.catch:
                self.params['ksdf'] = df_values.copy()
            
            if self.verbose:
                print("KS points computed.")
            return df_values

    def _generate_ks_points(self, N):
        """Generate Kolmogorov-Smirnov points."""
        if N <= 0:
            raise ValueError("N must be a positive integer.")
        
        n = np.arange(1, N + 1)
        ks_points = (2 * n - 1) / (2 * N)
        
        if self.catch:
            self.params['ks_points'] = ks_points.copy()
        
        return ks_points

    def _compute_egdf_core(self, S, LB, UB, zi_data=None, zi_eval=None):
        """Core EGDF computation with caching."""
        # Use provided data or default to instance data
        if zi_data is None:
            zi_data = self.z
        if zi_eval is None:
            zi_eval = zi_data
        
        # Convert to infinite domain
        zi_n = DataConversion._convert_fininf(zi_eval, LB, UB)
        zi_d = DataConversion._convert_fininf(zi_data, LB, UB)
        
        # Calculate R matrix with numerical stability
        R = zi_n.reshape(-1, 1) / (zi_d.reshape(1, -1) + self._NUMERICAL_EPS)
        
        # Get characteristics
        gc = GnosticsCharacteristics(R=R)
        q, q1 = gc._get_q_q1(S=S)
        
        # Calculate fidelities and irrelevances
        fi = gc._fi(q=q, q1=q1)
        hi = gc._hi(q=q, q1=q1)
        
        # Estimate EGDF
        return self._estimate_egdf_from_moments(fi, hi), fi, hi

    def _estimate_egdf_from_moments(self, fidelities, irrelevances):
        """Estimate EGDF from fidelities and irrelevances."""
        weights = self._computation_cache['weights_normalized'].reshape(-1, 1)
        
        mean_fidelity = np.sum(weights * fidelities, axis=0) / np.sum(weights)
        mean_irrelevance = np.sum(weights * irrelevances, axis=0) / np.sum(weights)
        
        M_zi = np.sqrt(mean_fidelity**2 + mean_irrelevance**2)
        M_zi = np.where(M_zi == 0, self._NUMERICAL_EPS, M_zi)
        
        egdf_values = (1 - mean_irrelevance / M_zi) / 2
        egdf_values = np.maximum.accumulate(egdf_values)
        egdf_values = np.clip(egdf_values, 0, 1)
        
        return egdf_values.flatten()

    # =============================================================================
    # OPTIMIZATION
    # =============================================================================
    
    def _determine_optimization_strategy(self):
        """Determine which parameters to optimize based on inputs."""
        s_is_auto = isinstance(self.S, str) and self.S.lower() == 'auto'
        lb_provided = self.LB is not None
        ub_provided = self.UB is not None
        
        if s_is_auto and not lb_provided and not ub_provided:
            # Optimize all parameters
            self.S_opt, self.LB_opt, self.UB_opt = self._optimize_all_parameters()
        elif lb_provided and ub_provided and s_is_auto:
            # Optimize only S
            self.LB_opt = self.LB
            self.UB_opt = self.UB
            self.S_opt = self._optimize_s_parameter(self.LB_opt, self.UB_opt)
        elif not s_is_auto and (not lb_provided or not ub_provided):
            # Optimize bounds only
            self.S_opt = self.S
            _, self.LB_opt, self.UB_opt = self._optimize_bounds_parameters(self.S_opt)
        else:
            # Use provided parameters
            self.S_opt = self.S if not s_is_auto else 1.0
            self.LB_opt = self.LB
            self.UB_opt = self.UB
        
        if self.verbose:
            print(f"Optimized parameters: S={self.S_opt:.6f}, LB={self.LB_opt:.6f}, UB={self.UB_opt:.6f}")

    def _optimize_all_parameters(self):
        """Optimize all parameters using normalized parameter space."""
        bounds = self._OPTIMIZATION_BOUNDS
        
        def normalize_params(s, lb, ub):
            s_norm = (s - bounds['S_MIN']) / (bounds['S_MAX'] - bounds['S_MIN'])
            lb_norm = (lb - bounds['LB_MIN']) / (bounds['LB_MAX'] - bounds['LB_MIN'])
            ub_norm = (ub - bounds['UB_MIN']) / (bounds['UB_MAX'] - bounds['UB_MIN'])
            return s_norm, lb_norm, ub_norm
        
        def denormalize_params(s_norm, lb_norm, ub_norm):
            s = bounds['S_MIN'] + s_norm * (bounds['S_MAX'] - bounds['S_MIN'])
            lb = bounds['LB_MIN'] + lb_norm * (bounds['LB_MAX'] - bounds['LB_MIN'])
            ub = bounds['UB_MIN'] + ub_norm * (bounds['UB_MAX'] - bounds['UB_MIN'])
            return s, lb, ub
        
        def objective_function(norm_params):
            try:
                s, lb, ub = denormalize_params(*norm_params)
                
                if s <= 0 or ub <= lb:
                    return 1e6
                
                egdf_values, _, _ = self._compute_egdf_core(s, lb, ub)
                diff = np.mean(np.abs(egdf_values - self.df_values) * self.weights)
                
                # Regularization
                reg = np.sum(np.array(norm_params)**2)
                
                total_loss = diff + reg
                
                if self.verbose:
                    print(f"Loss: {diff:.6f}, Total: {total_loss:.6f}, S: {s:.3f}, LB: {lb:.6f}, UB: {ub:.3f}")
                
                return total_loss
            except:
                return 1e6
        
        # Initial values
        s_init = 0.05
        lb_init = self.LB_init if hasattr(self, 'LB_init') and self.LB_init is not None else bounds['LB_MIN']
        ub_init = self.UB_init if hasattr(self, 'UB_init') and self.UB_init is not None else bounds['UB_MAX']
        
        initial_params = normalize_params(s_init, lb_init, ub_init)
        norm_bounds = [(0.0, 1.0)]
        
        try:
            result = minimize(
                objective_function,
                initial_params,
                method=self.opt_method,
                bounds=norm_bounds,
                options={'maxiter': 10000, 'ftol': self.tolerance},
                tol=self.tolerance  
            )
            
            s_opt, lb_opt, ub_opt = denormalize_params(*result.x)
            
            if lb_opt >= ub_opt:
                if self.verbose:
                    print("Warning: Optimized LB >= UB, using initial values")
                return s_init, lb_init, ub_init
            
            return s_opt, lb_opt, ub_opt
        except Exception as e:
            if self.verbose:
                print(f"Optimization failed: {e}")
            return s_init, lb_init, ub_init

    def _optimize_s_parameter(self, lb, ub):
        """Optimize only S parameter."""
        def objective_function(s):
            try:
                egdf_values, _, _ = self._compute_egdf_core(s[0], lb, ub)
                diff = np.mean(np.abs(egdf_values - self.df_values) * self.weights)
                if self.verbose:
                    print(f"S optimization - Loss: {diff:.6f}, S: {s[0]:.3f}")
                return diff
            except:
                return 1e6
        
        try:
            result = minimize(
                objective_function,
                [1.0],
                bounds=[(self._OPTIMIZATION_BOUNDS['S_MIN'], self._OPTIMIZATION_BOUNDS['S_MAX'])],
                method=self.opt_method,
                options={'maxiter': 1000}
            )
            return result.x[0]
        except:
            return 1.0

    def _optimize_bounds_parameters(self, s):
        """Optimize only LB and UB parameters."""
        bounds = self._OPTIMIZATION_BOUNDS
        
        def normalize_bounds(lb, ub):
            lb_norm = (lb - bounds['LB_MIN']) / (bounds['LB_MAX'] - bounds['LB_MIN'])
            ub_norm = (ub - bounds['UB_MIN']) / (bounds['UB_MAX'] - bounds['UB_MIN'])
            return lb_norm, ub_norm
        
        def denormalize_bounds(lb_norm, ub_norm):
            lb = bounds['LB_MIN'] + lb_norm * (bounds['LB_MAX'] - bounds['LB_MIN'])
            ub = bounds['UB_MIN'] + ub_norm * (bounds['UB_MAX'] - bounds['UB_MIN'])
            return lb, ub
        
        def objective_function(norm_params):
            try:
                lb, ub = denormalize_bounds(*norm_params)
                
                if lb <= 0 or ub <= lb:
                    return 1e6
                
                egdf_values, _, _ = self._compute_egdf_core(s, lb, ub)
                diff = np.mean(np.abs(egdf_values - self.df_values) * self.weights)
                
                # Regularization
                reg = np.sum(np.array(norm_params)**2)
                total_loss = diff + reg
                
                if self.verbose:
                    print(f"Bounds optimization - Loss: {diff:.6f}, Total: {total_loss:.6f}, LB: {lb:.6f}, UB: {ub:.3f}")
                
                return total_loss
            except:
                return 1e6
        
        # Initial values
        lb_init = self.LB_init if hasattr(self, 'LB_init') and self.LB_init is not None else bounds['LB_MIN']
        ub_init = self.UB_init if hasattr(self, 'UB_init') and self.UB_init is not None else bounds['UB_MIN']
        
        lb_init = np.clip(lb_init, bounds['LB_MIN'], bounds['LB_MAX'])
        ub_init = np.clip(ub_init, bounds['UB_MIN'], bounds['UB_MAX'])
        
        if lb_init >= ub_init:
            lb_init = bounds['LB_MIN']
            ub_init = bounds['UB_MIN']
        
        initial_params = normalize_bounds(lb_init, ub_init)
        norm_bounds = [(0.0, 1.0), (0.0, 1.0)]
        
        try:
            result = minimize(
                objective_function,
                initial_params,
                method=self.opt_method,
                bounds=norm_bounds,
                options={'maxiter': 10000, 'ftol': self.tolerance},
                tol=self.tolerance
            )
            
            lb_opt, ub_opt = denormalize_bounds(*result.x)
            
            if lb_opt >= ub_opt:
                if self.verbose:
                    print("Warning: Optimized LB >= UB, using initial values")
                return s, lb_init, ub_init
            
            return s, lb_opt, ub_opt
        except Exception as e:
            if self.verbose:
                print(f"Bounds optimization failed: {e}")
            return s, self.LB, self.UB


    # =============================================================================
    # PDF AND DERIVATIVE CALCULATIONS
    # =============================================================================
    
    # def _calculate_pdf_from_moments(self, fidelities, irrelevances):
    #     """Calculate PDF from fidelities and irrelevances."""
    #     weights = self._computation_cache['weights_normalized'].reshape(-1, 1)
        
    #     mean_fidelity = np.sum(weights * fidelities, axis=0) / np.sum(weights)
    #     mean_irrelevance = np.sum(weights * irrelevances, axis=0) / np.sum(weights)
        
    #     F2 = np.sum(weights * fidelities**2, axis=0) / np.sum(weights)
    #     FH = np.sum(weights * fidelities * irrelevances, axis=0) / np.sum(weights)
        
    #     M_zi = np.sqrt(mean_fidelity**2 + mean_irrelevance**2)
    #     M_zi = np.where(M_zi == 0, self._NUMERICAL_EPS, M_zi)
    #     M_zi_cubed = M_zi**3
        
    #     numerator = (mean_fidelity**2) * F2 + mean_fidelity * mean_irrelevance * FH
    #     S_value = self.S_opt if hasattr(self, 'S_opt') else 1.0
    #     density = (1 / S_value) * (numerator / M_zi_cubed)
        
    #     if np.any(density < 0):
    #         warnings.warn("PDF contains negative values, indicating potential non-homogeneous data", RuntimeWarning)
        
    #     return density.flatten()

    def _calculate_pdf_from_moments(self, fidelities, irrelevances): # PDF
        """Calculate first derivative of EGDF (which is the PDF) from stored fidelities and irrelevances."""
        if fidelities is None or irrelevances is None:
            raise ValueError("Fidelities and irrelevances must be calculated before first derivative estimation.")
        
        weights = self.weights.reshape(-1, 1)
        
        # First order moments
        f1 = np.sum(weights * fidelities, axis=0) / np.sum(weights)  # mean_fidelity
        h1 = np.sum(weights * irrelevances, axis=0) / np.sum(weights)  # mean_irrelevance

        # Second order moments (scaled by S as in MATLAB)
        f2s = np.sum(weights * (fidelities**2 / self.S_opt), axis=0) / np.sum(weights)
        fhs = np.sum(weights * (fidelities * irrelevances / self.S_opt), axis=0) / np.sum(weights)
        
        # Calculate denominator w = (f1^2 + h1^2)^(3/2)
        w = (f1**2 + h1**2)**(3/2)
        eps = np.finfo(float).eps
        w = np.where(w == 0, eps, w)
        
        # First derivative formula from MATLAB: y = (f1^2 * f2s + f1 * h1 * fhs) / w
        numerator = f1**2 * f2s + f1 * h1 * fhs
        first_derivative = numerator / w
        # first_derivative = first_derivative / self.zi
        
        # if np.any(first_derivative < 0):
        #     warnings.warn("EGDF first derivative (PDF) contains negative values, indicating potential non-homogeneous data", RuntimeWarning)
        return first_derivative.flatten()



    # =============================================================================
    # HOMOGENEITY TESTING
    # =============================================================================
    
    # def _test_data_homogeneity(self):
    #     """
    #     Test if the given data sample is homogeneous.
        
    #     Conditions for homogeneous data:
    #     1. PDF has single peak (one global maximum)
    #     2. PDF values are never negative
    #     """
    #     # Use PDF from smooth points if available, otherwise from data points
    #     if hasattr(self, 'pdf_points') and self.pdf_points is not None:
    #         pdf_to_test = self.pdf_points
    #     elif hasattr(self, 'pdf') and self.pdf is not None:
    #         pdf_to_test = self.pdf
    #     else:
    #         # Calculate PDF if not available
    #         if hasattr(self, 'fi') and hasattr(self, 'hi'):
    #             pdf_to_test = self._calculate_pdf_from_moments(self.fi, self.hi)
    #         else:
    #             if self.verbose:
    #                 print("Warning: Cannot test homogeneity - PDF not available")
    #             return {'is_homogeneous': None, 'error': 'PDF not available'}
        
    #     # Check for negative PDF values
    #     has_negative_pdf = np.any(pdf_to_test < 0)
        
    #     # Find peaks in PDF
    #     pdf_peaks = []
    #     for i in range(1, len(pdf_to_test) - 1):
    #         if pdf_to_test[i] > pdf_to_test[i-1] and pdf_to_test[i] > pdf_to_test[i+1]:
    #             pdf_peaks.append(i)
        
    #     # Check for single global maximum
    #     has_single_peak = len(pdf_peaks) <= 1
        
    #     # Overall homogeneity assessment
    #     is_homogeneous = not has_negative_pdf and has_single_peak
        
    #     homogeneity_results = {
    #         'is_homogeneous': is_homogeneous,
    #         'has_negative_pdf': has_negative_pdf,
    #         'number_of_peaks': len(pdf_peaks),
    #         'peak_locations': pdf_peaks,
    #         'has_single_peak': has_single_peak
    #     }
        
    #     return homogeneity_results


    def _calculate_final_results(self):
        """Calculate final EGDF and PDF with optimized parameters."""
        # Convert to infinite domain
        # zi_n = DataConversion._convert_fininf(self.z, self.LB_opt, self.UB_opt)
        zi_d = DataConversion._convert_fininf(self.z, self.LB_opt, self.UB_opt)
        self.zi = zi_d
        
        # Calculate EGDF and get moments
        egdf_values, fi, hi = self._compute_egdf_core(self.S_opt, self.LB_opt, self.UB_opt)
        
        # Store for derivative calculations
        self.fi = fi
        self.hi = hi
        self.egdf = egdf_values
        self.pdf = self._calculate_pdf_from_moments(fi, hi)
        
        if self.catch:
            self.params.update({
                'egdf': self.egdf.copy(),
                'pdf': self.pdf.copy(),
                'zi': self.zi.copy()
            })

    def _generate_smooth_curves(self):
        """Generate smooth curves for plotting and analysis."""
        try:
            # Generate smooth EGDF and PDF
            smooth_egdf, self.smooth_fi, self.smooth_hi = self._compute_egdf_core(
                self.S_opt, self.LB_opt, self.UB_opt,
                zi_data=self.z_points_n, zi_eval=self.z
            )
            
            smooth_pdf = self._calculate_pdf_from_moments(self.smooth_fi, self.smooth_hi)
            
            self.egdf_points = smooth_egdf
            self.pdf_points = smooth_pdf
            
            # Store zi_n for derivative calculations
            self.zi_n = DataConversion._convert_fininf(self.z_points_n, self.LB_opt, self.UB_opt)
            
            # Mark as generated
            self._computation_cache['smooth_curves_generated'] = True
            
            if self.catch:
                self.params.update({
                    'egdf_points': self.egdf_points.copy(),
                    'pdf_points': self.pdf_points.copy(),
                    'zi_points': self.zi_n.copy()
                })
            
            if self.verbose:
                print(f"Generated smooth curves with {self.n_points} points.")
                
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not generate smooth curves: {e}")
            # Create fallback points using original data
            self.egdf_points = self.egdf.copy() if hasattr(self, 'egdf') else None
            self.pdf_points = self.pdf.copy() if hasattr(self, 'pdf') else None
            self._computation_cache['smooth_curves_generated'] = False

    def _transform_bounds_to_original_domain(self):
        """Transform optimized bounds back to original domain."""
        dc = self._get_data_converter()
        
        if self.data_form == 'a':
            self.LB = dc._convert_za(self.LB_opt, self.DLB, self.DUB)
            self.UB = dc._convert_za(self.UB_opt, self.DLB, self.DUB)
        else:
            self.LB = dc._convert_zm(self.LB_opt, self.DLB, self.DUB)
            self.UB = dc._convert_zm(self.UB_opt, self.DLB, self.DUB)
        
        if self.catch:
            self.params.update({'LB': float(self.LB), 'UB': float(self.UB), 'S_opt': float(self.S_opt)})

    # =============================================================================
    # PLOTTING FUNCTIONALITY - IMPROVED
    # =============================================================================
    
    def _plot(self, plot_smooth: bool = True, plot: str = 'both', bounds: bool = True, extra_df: bool = True, figsize: tuple = (12, 8)):
        """Enhanced plotting with better organization."""
        import matplotlib.pyplot as plt

        if plot_smooth and (len(self.data) > self.max_data_size) and self.verbose:
            print(f"Warning: Given data size ({len(self.data)}) exceeds max_data_size ({self.max_data_size}). For optimal compute performance, set 'plot_smooth=False', or 'max_data_size' to a larger value whichever is appropriate.")

        if not self.catch:
            print("Plot is not available with argument catch=False")
            return
        
        if not self._fitted:
            raise RuntimeError("Must fit EGDF before plotting.")
        
        # Validate plot parameter
        if plot not in ['gdf', 'pdf', 'both']:
            raise ValueError("plot parameter must be 'gdf', 'pdf', or 'both'")
        
        # Check data availability
        if plot in ['gdf', 'both'] and self.params.get('egdf') is None:
            raise ValueError("EGDF must be calculated before plotting GDF")
        if plot in ['pdf', 'both'] and self.params.get('pdf') is None:
            raise ValueError("PDF must be calculated before plotting PDF")
        
        # Prepare data
        x_points = self.data
        egdf_plot = self.params.get('egdf')
        pdf_plot = self.params.get('pdf')
        wedf = self.params.get('wedf')
        ksdf = self.params.get('ksdf')
        
        # Check smooth plotting availability
        has_smooth = (hasattr(self, 'di_points_n') and hasattr(self, 'egdf_points') 
                     and hasattr(self, 'pdf_points') and self.di_points_n is not None
                     and self.egdf_points is not None and self.pdf_points is not None)
        plot_smooth = plot_smooth and has_smooth
        
        # Create figure
        fig, ax1 = plt.subplots(figsize=figsize)

        # Plot EGDF if requested
        if plot in ['gdf', 'both']:
            self._plot_egdf(ax1, x_points, egdf_plot, plot_smooth, extra_df, wedf, ksdf)
        
        # Plot PDF if requested
        if plot in ['pdf', 'both']:
            if plot == 'pdf':
                self._plot_pdf(ax1, x_points, pdf_plot, plot_smooth, is_secondary=False)
            else:
                ax2 = ax1.twinx()
                self._plot_pdf(ax2, x_points, pdf_plot, plot_smooth, is_secondary=True)
        
        # Add bounds and formatting
        self._add_plot_formatting(ax1, plot, bounds)
        
        plt.tight_layout()
        plt.show()

    def _plot_egdf(self, ax, x_points, egdf_plot, plot_smooth, extra_df, wedf, ksdf):
        """Plot EGDF components."""
        if plot_smooth and hasattr(self, 'egdf_points') and self.egdf_points is not None:
            ax.plot(x_points, egdf_plot, 'o', color='blue', label='EGDF', markersize=4)
            ax.plot(self.di_points_n, self.egdf_points, color='blue', 
                   linestyle='-', linewidth=2, alpha=0.8)
        else:
            ax.plot(x_points, egdf_plot, 'o-', color='blue', label='EGDF', 
                   markersize=4, linewidth=1, alpha=0.8)
        
        if extra_df:
            if wedf is not None:
                ax.plot(x_points, wedf, 's', color='lightblue', 
                       label='WEDF', markersize=3, alpha=0.8)
            if ksdf is not None:
                ax.plot(x_points, ksdf, 's', color='cyan', 
                       label='KS Points', markersize=3, alpha=0.8)
        
        ax.set_ylabel('EGDF', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax.set_ylim(0, 1)

    def _plot_pdf(self, ax, x_points, pdf_plot, plot_smooth, is_secondary=False):
        """Plot PDF components."""
        color = 'red'

        if plot_smooth and hasattr(self, 'pdf_points') and self.pdf_points is not None:
            ax.plot(x_points, pdf_plot, 'o', color=color, label='PDF', markersize=4)
            ax.plot(self.di_points_n, self.pdf_points, color=color, 
                   linestyle='-', linewidth=2, alpha=0.8)
            max_pdf = np.max(self.pdf_points)
        else:
            ax.plot(x_points, pdf_plot, 'o-', color=color, label='PDF', 
                   markersize=4, linewidth=1, alpha=0.8)
            max_pdf = np.max(pdf_plot)
        
        ax.set_ylabel('PDF', color=color)
        ax.tick_params(axis='y', labelcolor=color)
        ax.set_ylim(0, max_pdf * 1.1)
        
        if is_secondary:
            ax.legend(loc='upper right', bbox_to_anchor=(1, 1))

    def _add_plot_formatting(self, ax1, plot, bounds):
        """Add formatting, bounds, and legends to plot."""
        ax1.set_xlabel('Data Points')
        
        # Add bounds if requested
        if bounds:
            bound_info = [
                (self.params.get('DLB'), 'green', '-', 'DLB'),
                (self.params.get('DUB'), 'orange', '-', 'DUB'),
                (self.params.get('LB'), 'purple', '--', 'LB'),
                (self.params.get('UB'), 'brown', '--', 'UB')
            ]
            
            for bound, color, style, name in bound_info:
                if bound is not None:
                    ax1.axvline(x=bound, color=color, linestyle=style, linewidth=2, 
                               alpha=0.8, label=f"{name}={bound:.3f}")
            
            # Add shaded regions
            if self.params.get('LB') is not None:
                ax1.axvspan(self.data.min(), self.params['LB'], alpha=0.15, color='purple')
            if self.params.get('UB') is not None:
                ax1.axvspan(self.params['UB'], self.data.max(), alpha=0.15, color='brown')
        
        # Set limits and add grid
        data_range = self.params['DUB'] - self.params['DLB']
        padding = data_range * 0.1
        ax1.set_xlim(self.params['DLB'] - padding, self.params['DUB'] + padding)
        
        # Set title
        titles = {
            'gdf': 'EGDF' + (' with Bounds' if bounds else ''),
            'pdf': 'PDF' + (' with Bounds' if bounds else ''),
            'both': 'EGDF and PDF' + (' with Bounds' if bounds else '')
        }
        
        ax1.set_title(titles[plot])
        ax1.legend(loc='upper left', bbox_to_anchor=(0, 1))
        ax1.grid(True, alpha=0.3)



    # =============================================================================
    # CLEANUP AND MEMORY MANAGEMENT
    # =============================================================================
    
    def _cleanup_computation_cache(self):
        """Clean up temporary computation cache to free memory."""
        self._computation_cache = {
            'data_converter': None,
            'characteristics_computer': None,
            'weights_normalized': None,
            'smooth_curves_generated': False
        }
        
        # Remove large temporary arrays if they exist
        temp_attrs = ['fi', 'hi', 'df_values']
        for attr in temp_attrs:
            if hasattr(self, attr):
                delattr(self, attr)

        # # delet di_points_n, z_points_n, egdf_points, pdf_points, zi_n if they exist
        # if hasattr(self, 'egdf_points'):
        #     del self.egdf_points
        # if hasattr(self, 'pdf_points'):
        #     del self.pdf_points
        # if hasattr(self, 'di_points_n'):
        #     del self.di_points_n
        # if hasattr(self, 'z_points_n'):
        #     del self.z_points_n
        # if hasattr(self, 'zi_n'):
        #     del self.zi_n
        # if self.catch:
        #     self.params.update({
        #         'egdf_points': None,
        #         'pdf_points': None,
        #         'di_points_n': None,
        #         'z_points_n': None,
        #         'zi_n': None
        #     })
        # deleting long arrays from params like z_points, di_points
        long_array_params = ['z_points', 'di_points', 'egdf_points', 'pdf_points', 'zi_n', 'zi_points']
        for param in long_array_params:
            if param in self.params:
                self.params[param] = None

        if self.verbose:
            print("Computation cache cleaned up.")


    # =============================================================================
    # Derivative
    # =============================================================================
    def _get_egdf_second_derivative(self):
        """Calculate second derivative of EGDF from stored fidelities and irrelevances."""
        if self.fi is None or self.hi is None:
            raise ValueError("Fidelities and irrelevances must be calculated before second derivative estimation.")
        
        weights = self.weights.reshape(-1, 1)
        
        # Moment calculations
        f1 = np.sum(weights * self.fi, axis=0) / np.sum(weights)
        h1 = np.sum(weights * self.hi, axis=0) / np.sum(weights)
        f2 = np.sum(weights * self.fi**2, axis=0) / np.sum(weights)
        f3 = np.sum(weights * self.fi**3, axis=0) / np.sum(weights)
        fh = np.sum(weights * self.fi * self.hi, axis=0) / np.sum(weights)
        fh2 = np.sum(weights * self.fi * self.hi**2, axis=0) / np.sum(weights)
        f2h = np.sum(weights * self.fi**2 * self.hi, axis=0) / np.sum(weights)
        
        # Calculate components
        b = f1**2 * f2 + f1 * h1 * fh
        d = f1**2 + h1**2
        eps = np.finfo(float).eps
        d = np.where(d == 0, eps, d)
        
        # Following
        term1 = f1 * (h1 * (f3 - fh2) - f2 * fh)
        term2 = 2 * f1**2 * f2h + h1 * fh**2
        term3 = (6 * b * (f1 * fh - h1 * f2)) / d
        
        d2 = -1 / (d**(1.5)) * (2 * (term1 - term2) + term3)
        second_derivative = d2 / (self.S_opt**2)
        # second_derivative = second_derivative / self.zi**2 
        return second_derivative.flatten()

    def _get_egdf_third_derivative(self):
        """Calculate third derivative of EGDF from stored fidelities and irrelevances."""
        if self.fi is None or self.hi is None:
            raise ValueError("Fidelities and irrelevances must be calculated before third derivative estimation.")
        
        weights = self.weights.reshape(-1, 1)
        
        # All required moments
        f1 = np.sum(weights * self.fi, axis=0) / np.sum(weights)
        h1 = np.sum(weights * self.hi, axis=0) / np.sum(weights)
        f2 = np.sum(weights * self.fi**2, axis=0) / np.sum(weights)
        f3 = np.sum(weights * self.fi**3, axis=0) / np.sum(weights)
        f4 = np.sum(weights * self.fi**4, axis=0) / np.sum(weights)
        fh = np.sum(weights * self.fi * self.hi, axis=0) / np.sum(weights)
        h2 = np.sum(weights * self.hi**2, axis=0) / np.sum(weights)
        fh2 = np.sum(weights * self.fi * self.hi**2, axis=0) / np.sum(weights)
        f2h = np.sum(weights * self.fi**2 * self.hi, axis=0) / np.sum(weights)
        f2h2 = np.sum(weights * self.fi**2 * self.hi**2, axis=0) / np.sum(weights)
        f3h = np.sum(weights * self.fi**3 * self.hi, axis=0) / np.sum(weights)
        fh3 = np.sum(weights * self.fi * self.hi**3, axis=0) / np.sum(weights)
        
        # Following
        # Derivative calculations
        dh1 = -f2
        df1 = fh
        df2 = 2 * f2h
        dfh = -f3 + fh2
        dfh2 = -2 * f3h + fh3
        df3 = 3 * f3h
        df2h = -f4 + 2 * f2h2
        
        # u4 and its derivative
        u4 = h1 * f3 - h1 * fh2 - f2 * fh
        du4 = dh1 * f3 + h1 * df3 - dh1 * fh2 - h1 * dfh2 - df2 * fh - f2 * dfh
        
        # u and its derivative
        u = f1 * u4
        du = df1 * u4 + f1 * du4
        
        # v components
        v4a = (f1**2) * f2h
        dv4a = 2 * f1 * df1 * f2h + (f1**2) * df2h
        v4b = h1 * fh**2
        dv4b = dh1 * (fh**2) + 2 * h1 * fh * dfh
        
        v = 2 * v4a + v4b
        dv = 2 * dv4a + dv4b
        
        # x components
        x4a = f1**2 * f2 + f1 * h1 * fh
        dx4a = 2 * f1 * df1 * f2 + (f1**2) * df2 + df1 * h1 * fh + f1 * dh1 * fh + f1 * h1 * dfh
        x4b = f1 * fh - h1 * f2
        dx4b = df1 * fh + f1 * dfh - dh1 * f2 - h1 * df2
        
        x = 6 * x4a * x4b
        dx = 6 * (dx4a * x4b + x4a * dx4b)
        
        # d components
        d = f1**2 + h1**2
        dd = 2 * (f1 * df1 + h1 * dh1)
        eps = np.finfo(float).eps
        d = np.where(d == 0, eps, d)
        
        # Final calculation
        term1 = (du - dv) / (d**1.5) - (1.5 * (u - v)) / (d**2.5) * dd
        term2 = dx / (d**2.5) - (2.5 * x) / (d**3.5) * dd
        
        d3p = -2 * term1 - term2
        third_derivative = 2 * d3p / (self.S_opt**3)
        # third_derivative = third_derivative / (self.zi**3)
        return third_derivative.flatten()

    def _get_egdf_fourth_derivative(self):
        """Calculate fourth derivative of EGDF using numerical differentiation."""
        if self.fi is None or self.hi is None:
            raise ValueError("Fidelities and irrelevances must be calculated before fourth derivative estimation.")
        
        # For fourth derivative, use numerical differentiation as it's complex
        dz = 1e-7
        
        # Get third derivatives at slightly shifted points
        zi_plus = self.zi + dz
        zi_minus = self.zi - dz
        
        # Store original zi
        original_zi = self.zi.copy()
        
        # Calculate third derivative at zi + dz
        self.zi = zi_plus
        self._calculate_fidelities_irrelevances_at_given_zi(self.zi)
        third_plus = self._get_egdf_third_derivative()
        
        # Calculate third derivative at zi - dz  
        self.zi = zi_minus
        self._calculate_fidelities_irrelevances_at_given_zi(self.zi)
        third_minus = self._get_egdf_third_derivative()
        
        # Restore original zi and recalculate fi, hi
        self.zi = original_zi
        self._calculate_fidelities_irrelevances_at_given_zi(self.zi)
        
        # Numerical derivative
        fourth_derivative = (third_plus - third_minus) / (2 * dz) * self.zi
        
        return fourth_derivative.flatten()

    def _calculate_fidelities_irrelevances_at_given_zi(self, zi):
        """Helper method to recalculate fidelities and irrelevances for current zi."""
        # Convert to infinite domain
        zi_n = DataConversion._convert_fininf(self.z, self.LB_opt, self.UB_opt)
        # is zi given then use it, else use self.zi
        if zi is None:
            zi_d = self.zi
        else:
            zi_d = zi

        # Calculate R matrix
        eps = np.finfo(float).eps
        R = zi_n.reshape(-1, 1) / (zi_d + eps).reshape(1, -1)

        # Get characteristics
        gc = GnosticsCharacteristics(R=R)
        q, q1 = gc._get_q_q1(S=self.S_opt)
        
        # Store fidelities and irrelevances
        self.fi = gc._fi(q=q, q1=q1)
        self.hi = gc._hi(q=q, q1=q1)


    # =============================================================================
    # MAIN FITTING PROCESS
    # =============================================================================
    
    def _fit_egdf(self):
        """Main fitting process with improved organization."""
        if self.verbose:
            print("Starting EGDF fitting process...")
        
        # Step 1: Data preprocessing
        self.data = np.sort(self.data)
        self._estimate_data_bounds()
        self._transform_data_to_standard_domain()
        self._estimate_weights()
        
        # Step 2: Bounds estimation
        self._estimate_initial_probable_bounds()
        self._generate_evaluation_points()
        
        # Step 3: Get distribution function values for optimization
        self.df_values = self._get_distribution_function_values(use_wedf=self.wedf)
        
        # Step 4: Parameter optimization
        self._determine_optimization_strategy()
        
        # Step 5: Calculate final EGDF and PDF
        self._calculate_final_results()
        
        # Step 6: Generate smooth curves for plotting and analysis
        self._generate_smooth_curves()
        
        # Step 7: Transform bounds back to original domain
        self._transform_bounds_to_original_domain()
        
        # Mark as fitted (Step 8 is now optional via marginal_analysis())
        self._fitted = True
        
        if self.verbose:
            print("EGDF fitting completed successfully.")
        
        # clean up computation cache
        if self.flush:  
            self._cleanup_computation_cache()