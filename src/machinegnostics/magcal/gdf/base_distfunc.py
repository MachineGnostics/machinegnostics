'''
Base Compute class for GDF

Author: Nirmal Parmar
Machine Gnostics
'''

import numpy as np
import warnings
from scipy.optimize import minimize
from machinegnostics.magcal.characteristics import GnosticsCharacteristics
from machinegnostics.magcal.gdf.base_df import BaseDistFunc
from machinegnostics.magcal.data_conversion import DataConversion
from machinegnostics.magcal.gdf.wedf import WEDF
from machinegnostics.magcal.mg_weights import GnosticsWeights

class BaseDistFuncCompute(BaseDistFunc):
    '''Base Distribution Function class
    Base class for EGDF (Estimating Global Distribution Function).
    
    This class provides a comprehensive framework for estimating global distribution
    functions with optimization capabilities and derivative analysis.
    '''
    
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

    def __init__(self,
                 data: np.ndarray,
                 DLB: float = None,
                 DUB: float = None,
                 LB: float = None,
                 UB: float = None,
                 S = 'auto',
                 varS: bool = False,
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
        self.varS = varS
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
        if not isinstance(self.varS, bool):
            raise ValueError("varS must be a boolean value. VarS can be only true for 'ELDF' and 'QLDF'.")
        
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
    
    def _determine_optimization_strategy(self):
        """Determine which parameters to optimize based on inputs."""
        if self.verbose:
            print("Determining optimization strategy...")
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
        if self.verbose:
            print("Optimizing all parameters (S, LB, UB)...")
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
        if self.verbose:
            print("Optimizing S parameter...")

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
        if self.verbose:
            print("Optimizing LB and UB parameters...")
            
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

