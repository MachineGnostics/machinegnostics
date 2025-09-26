"""
Distribution Function Parameter Optimization Engine

Internal optimization engine for ManGo distribution function parameter tuning.
This module provides a unified, robust optimization framework for EGDF, QGDF,
and other distribution function implementations.

Key Design Decisions:
- Normalized parameter space for numerical stability
- Strategy-based optimization (auto-detect what needs optimization)
- Comprehensive error handling with fallback mechanisms
- Modular design for easy extension to new distribution functions

Performance Considerations:
- Uses scipy.optimize.minimize with L-BFGS-B for bounded optimization
- Parameter normalization prevents scale-related convergence issues
- Caching and result storage for debugging optimization failures

Author: Nirmal Parmar
Machine Gnostics
"""

import logging
import numpy as np
from typing import Callable, Dict, Any, Tuple, Optional, Union
from scipy.optimize import minimize
from machinegnostics.magcal.util.logging import get_logger

class DistFuncEngine:
    """
    Internal optimization engine for distribution function parameters (S, LB, UB).
    
    DEVELOPER NOTES:
    ================
    
    Design Philosophy:
    ------------------
    This class was designed to replace scattered optimization code in EGDF and QGDF
    classes. The main goals were:
    1. Centralize parameter optimization logic
    2. Provide consistent behavior across different distribution functions
    3. Handle edge cases and optimization failures gracefully
    4. Enable debugging of optimization issues through comprehensive logging
    
    Parameter Space Design:
    -----------------------
    The optimization uses normalized parameter space [0,1]³ to avoid numerical
    issues with vastly different scales:
    - S: typically 0.05 to 100 (factor of 2000x difference)
    - LB: typically 1e-6 to exp(-1) (factor of ~300,000x difference)  
    - UB: typically exp(1) to 1e6 (factor of ~300,000x difference)
    
    Normalization prevents optimizer from getting stuck due to poor scaling.
    
    Optimization Strategies:
    ------------------------
    The engine automatically detects which parameters need optimization:
    
    1. 'optimize_all': S='auto', LB=None, UB=None
       - Most computationally expensive
       - Uses 3D normalized optimization with regularization
       - Fallback to reasonable defaults if optimization fails
    
    2. 'optimize_s_only': S='auto', LB=provided, UB=provided
       - Fast 1D optimization over S parameter
       - Bounds are fixed to user-provided values
       - Common case when bounds are known from data analysis
    
    3. 'optimize_bounds_only': S=provided, LB=None OR UB=None
       - 2D optimization over bounds with fixed S
       - Less common, used when S is known from theory/experiment
    
    4. 'use_provided': All parameters explicitly provided
       - No optimization, direct parameter assignment
       - Used for validation runs or when parameters are pre-computed
    
    Error Handling Strategy:
    ------------------------
    Multiple layers of error handling:
    1. Input validation at initialization (fail fast)
    2. Optimization-level error catching (graceful degradation)
    3. Objective function error catching (return penalty value)
    4. Fallback parameter estimation (always return something reasonable)
    
    This design ensures the engine never completely fails, which is critical
    for batch processing large datasets.
    
    Memory and Performance:
    -----------------------
    - Stores optimization results for debugging (can be disabled in production)
    - Error/warning logs are accumulated (consider periodic clearing for long runs)
    - Objective function evaluations are not cached (compute_func should handle this)
    - Uses scipy's L-BFGS-B which is memory-efficient for bounded optimization
    
    Thread Safety:
    --------------
    This class is NOT thread-safe. Each thread should have its own instance.
    The compute_func parameter may have its own thread-safety requirements.
    
    Integration Points:
    -------------------
    The compute_func parameter is the main integration point. It should:
    - Accept (s: float, lb: float, ub: float) parameters
    - Return (dist_values: np.ndarray, info1: Any, info2: Any) tuple
    - Handle numerical edge cases gracefully
    - Be reasonably efficient (called many times during optimization)
    
    Common Integration Examples:
    - compute_func = lambda s, lb, ub: self._compute_egdf_core(s, lb, ub)
    - compute_func = lambda s, lb, ub: self._compute_qgdf_core(s, lb, ub)
    
    Debugging Optimization Issues:
    ------------------------------
    Use get_optimization_info() to access:
    - Optimization convergence details (success, iterations, final objective value)
    - Error logs with full exception information
    - Parameter values attempted during optimization
    
    Common failure modes:
    1. Objective function numerical issues -> Check compute_func implementation
    2. Poor initial parameter guesses -> Adjust LB_init, UB_init
    3. Target values incompatible with parameter bounds -> Check data preprocessing
    4. Optimization stuck in local minimum -> Try different opt_method or tolerance
    
    Extension Guidelines:
    ---------------------
    To add new optimization features:
    1. Add new strategy to _determine_optimization_strategy()
    2. Implement corresponding _optimize_xxx_parameters() method
    3. Update optimize() method to handle new strategy
    4. Add appropriate fallback behavior
    5. Update optimization bounds if needed
    
    KNOWN LIMITATIONS:
    ==================
    1. No support for constrained optimization beyond bounds
    2. No automatic hyperparameter tuning (tolerance, max_iterations)
    3. No parallel optimization (could be added for multiple starting points)
    4. Regularization weight is global (could be parameter-specific)
    5. No adaptive optimization strategy based on problem characteristics
    
    TODO/FUTURE IMPROVEMENTS:
    =========================
    1. Add support for custom bounds per parameter
    2. Implement multi-start optimization for better global convergence
    3. Add automatic hyperparameter tuning based on problem size
    4. Consider using more advanced optimizers (e.g., differential evolution)
    5. Add optimization warm-starting from previous results
    """
    
    # Class constants for optimization bounds
    # These bounds are derived from practical experience with ManGo data
    _OPTIMIZATION_BOUNDS = {
        'S_MIN': 0.05,      # Below this, numerical instability in distribution functions
        'S_MAX': 100.0,     # Above this, diminishing returns and numerical issues
        'LB_MIN': 1e-6,     # Practical lower limit for meaningful bounds
        'LB_MAX': np.exp(-1.000001),  # Slightly less than e^-1 to avoid edge case
        'UB_MIN': np.exp(1.000001),   # Slightly more than e^1 to avoid edge case  
        'UB_MAX': 1e6,      # Practical upper limit for meaningful bounds
        'REGULARIZATION_WEIGHT': 1e-6  # Prevents overfitting to noise in target_values
    }
    
    # Numerical constants for fallback behavior
    _NUMERICAL_EPS = np.finfo(float).eps
    _FALLBACK_VALUES = {
        'S': 1.0,           # Neutral S value (no excessive sharpening/smoothing)
        'LB_FACTOR': 0.1,   # Conservative lower bound estimation
        'UB_FACTOR': 10.0   # Conservative upper bound estimation
    }

    def __init__(self,
                 compute_func: Callable,
                 target_values: np.ndarray,
                 weights: np.ndarray = None,
                 S: Union[float, str] = 'auto',
                 LB: float = None,
                 UB: float = None,
                 LB_init: float = None,
                 UB_init: float = None,
                 tolerance: float = 1e-3,
                 opt_method: str = 'L-BFGS-B',
                 max_iterations: int = 10000,
                 regularization_weight: float = None,
                 verbose: bool = False,
                 catch_errors: bool = True):
        """
        Initialize the Distribution Function Optimization Engine.
        
        DEVELOPER PARAMETERS:
        =====================
        
        compute_func : Callable
            Core distribution function to optimize. Must have signature:
            compute_func(s: float, lb: float, ub: float) -> Tuple[np.ndarray, Any, Any]
            
            The function should:
            - Return distribution values as first element of tuple
            - Handle edge cases (s<=0, lb>=ub, etc.) gracefully
            - Be numerically stable across the optimization bounds
            - Execute reasonably fast (called 100s-1000s of times)
            
            Example: lambda s, lb, ub: self._compute_egdf_core(s, lb, ub)
            
        target_values : np.ndarray
            Target distribution values to match (e.g., WEDF values, KS points).
            These are typically:
            - WEDF values for EGDF optimization
            - Kolmogorov-Smirnov points for theoretical comparisons
            - Pre-computed reference distribution values
            
            Must be same length as the output of compute_func.
            
        weights : np.ndarray, optional
            Point-wise weights for loss computation. If None, uniform weights used.
            Useful for:
            - Emphasizing tail behavior (higher weights at extremes)
            - Handling heteroscedastic data
            - Incorporating measurement uncertainties
            
        S : Union[float, str], default 'auto'
            Sharpening parameter. Options:
            - 'auto': Optimize S automatically
            - float > 0: Use fixed S value
            
            Typical values:
            - S < 1: Smoothing effect
            - S = 1: Neutral
            - S > 1: Sharpening effect
            
        LB, UB : float, optional
            Distribution bounds. If None, will be optimized.
            These should correspond to the transformed data domain.
            
            Critical: LB must be < UB if both provided
            
        LB_init, UB_init : float, optional
            Initial guesses for bound optimization. Good initial guesses
            significantly improve convergence:
            - LB_init: slightly below data minimum
            - UB_init: slightly above data maximum
            
        tolerance : float, default 1e-3
            Optimization convergence tolerance. Smaller values give more
            precise results but take longer to converge.
            
            Recommended values:
            - 1e-2: Fast, acceptable for most applications
            - 1e-3: Standard precision
            - 1e-4: High precision, slower convergence
            
        opt_method : str, default 'L-BFGS-B'
            Scipy optimization method. L-BFGS-B is recommended for bounded
            problems with gradients. Alternatives:
            - 'TNC': Alternative bounded optimizer
            - 'SLSQP': Sequential least squares (slower but robust)
            
        max_iterations : int, default 10000
            Maximum optimization iterations. Increase for difficult problems,
            decrease for faster (but potentially less accurate) optimization.
            
        regularization_weight : float, optional
            Weight for L2 regularization term. Helps prevent overfitting
            to noisy target values. Default is usually appropriate.
            
        verbose : bool, default False
            Enable detailed optimization logging. Useful for debugging
            but can generate significant output for large optimization runs.
            
        catch_errors : bool, default True
            Whether to catch optimization errors and return fallback values.
            - True: Always returns reasonable parameters (recommended for production)
            - False: Raises exceptions for debugging optimization issues
            
        INTERNAL STATE INITIALIZED:
        ===========================
        - optimization_results: Dict storing convergence information
        - optimization_errors: List of encountered errors with full context
        - optimization_warnings: List of non-critical issues
        - S_opt, LB_opt, UB_opt: Optimized parameter values (None until optimize() called)
        """
        
        # Validate inputs
        self._validate_inputs(compute_func, target_values, weights, S, LB, UB)
        
        # Store parameters
        self.compute_func = compute_func
        self.target_values = np.asarray(target_values)
        self.weights = weights if weights is not None else np.ones_like(self.target_values)
        self.S = S
        self.LB = LB
        self.UB = UB
        self.LB_init = LB_init
        self.UB_init = UB_init
        self.tolerance = tolerance
        self.opt_method = opt_method
        self.max_iterations = max_iterations
        self.regularization_weight = (regularization_weight if regularization_weight is not None 
                                    else self._OPTIMIZATION_BOUNDS['REGULARIZATION_WEIGHT'])
        self.verbose = verbose
        self.catch_errors = catch_errors
        
        # Results storage
        self.optimization_results = {}
        self.optimization_errors = []
        self.optimization_warnings = []
        
        # Optimized parameters
        self.S_opt = None
        self.LB_opt = None
        self.UB_opt = None

        # logger
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if verbose else logging.WARNING)
        self.logger.debug(f"{self.__class__.__name__} initialized:")
        
    def _validate_inputs(self, compute_func, target_values, weights, S, LB, UB):
        """Validate initialization inputs."""
        if not callable(compute_func):
            self.logger.error("compute_func must be callable")
            raise TypeError("compute_func must be callable")
            
        if not isinstance(target_values, np.ndarray):
            self.logger.debug("Converting target_values to numpy array.")
            target_values = np.asarray(target_values)
            
        if target_values.size == 0:
            self.logger.error("target_values cannot be empty")
            raise ValueError("target_values cannot be empty")
            
        if not np.isfinite(target_values).all():
            self.logger.error("target_values must contain only finite values")
            raise ValueError("target_values must contain only finite values")
            
        if weights is not None:
            weights = np.asarray(weights)
            if len(weights) != len(target_values):
                self.logger.error("weights must have same length as target_values")
                raise ValueError("weights must have same length as target_values")
            if not np.all(weights >= 0):
                self.logger.error("weights must be non-negative")
                raise ValueError("weights must be non-negative")
                
        # Validate S parameter
        if isinstance(S, str) and S.lower() != 'auto':
            self.logger.error("S must be a positive number or 'auto'")
            raise ValueError("S must be a positive number or 'auto'")
        elif isinstance(S, (int, float)) and S <= 0:
            self.logger.error("S must be positive when specified as a number")
            raise ValueError("S must be positive when specified as a number")
            
        # Validate bounds
        if LB is not None and UB is not None and LB >= UB:
            self.logger.error("LB must be less than UB when both are provided")
            raise ValueError("LB must be less than UB when both are provided")

    def optimize(self) -> Dict[str, float]:
        """
        Main optimization method. Determines strategy and optimizes parameters.
        
        Returns:
        --------
        Dict[str, float]
            Optimized parameters {'S': value, 'LB': value, 'UB': value}
        """
        self.logger.debug("Starting parameter optimization...")
            
        try:
            # Determine optimization strategy
            strategy = self._determine_optimization_strategy()

            self.logger.debug(f"Using optimization strategy: {strategy}")

            # Execute optimization based on strategy
            if strategy == 'optimize_all':
                self.S_opt, self.LB_opt, self.UB_opt = self._optimize_all_parameters()
            elif strategy == 'optimize_s_only':
                self.S_opt = self._optimize_s_parameter(self.LB_init, self.UB_init)
                self.LB_opt = self.LB_init
                self.UB_opt = self.UB_init
            elif strategy == 'optimize_bounds_only':
                _, self.LB_opt, self.UB_opt = self._optimize_bounds_parameters(self.S)
                self.S_opt = self.S
            else:  # use_provided
                self.S_opt = self.S if isinstance(self.S, (int, float)) else self._FALLBACK_VALUES['S']
                self.LB_opt = self.LB_init
                self.UB_opt = self.UB_init
            
            # Store results
            results = {
                'S': float(self.S_opt),
                'LB': float(self.LB_opt), 
                'UB': float(self.UB_opt)
            }
            
            self.optimization_results.update(results)
            self.optimization_results['strategy_used'] = strategy

            self.logger.info("Optimization completed successfully")
            self.logger.debug(f"Results - S: {self.S_opt:.6f}, LB: {self.LB_opt:.6f}, UB: {self.UB_opt:.6f}")

            return results
            
        except Exception as e:
            error_msg = f"DistFuncEngine optimization failed: {str(e)}"
            self.logger.error(error_msg)
            self.optimization_errors.append({
                'method': 'optimize',
                'error': error_msg,
                'exception_type': type(e).__name__
            })
            
            if self.catch_errors:
                if self.verbose:
                    self.logger.debug(f"{error_msg}")
                    self.logger.debug("Using fallback values")
                return self._get_fallback_results()
            else:
                raise

    def _determine_optimization_strategy(self) -> str:
        """Determine which optimization strategy to use."""
        self.logger.info("Determining optimization strategy...")
        s_is_auto = isinstance(self.S, str) and self.S.lower() == 'auto'
        lb_provided = self.LB is not None
        ub_provided = self.UB is not None
        
        if s_is_auto and not lb_provided and not ub_provided:
            return 'optimize_all'
        elif lb_provided and ub_provided and s_is_auto:
            # Set initial values from provided bounds
            self.LB_init = self.LB_init
            self.UB_init = self.UB_init
            return 'optimize_s_only'
        elif not s_is_auto and (not lb_provided or not ub_provided):
            return 'optimize_bounds_only'
        else:
            # All parameters provided
            self.LB_init = self.LB_init
            self.UB_init = self.UB_init
            return 'use_provided'

    def _optimize_all_parameters(self) -> Tuple[float, float, float]:
        """Optimize all parameters (S, LB, UB) using normalized parameter space."""
        self.logger.info("Optimizing all parameters (S, LB, UB)...")
            
        bounds = self._OPTIMIZATION_BOUNDS
        
        def normalize_params(s, lb, ub):
            s_norm = (s - bounds['S_MIN']) / (bounds['S_MAX'] - bounds['S_MIN'])
            lb_norm = (lb - bounds['LB_MIN']) / (bounds['LB_MAX'] - bounds['LB_MIN'])
            ub_norm = (ub - bounds['UB_MIN']) / (bounds['UB_MAX'] - bounds['UB_MIN'])
            return np.array([s_norm, lb_norm, ub_norm])
        
        def denormalize_params(norm_params):
            s_norm, lb_norm, ub_norm = norm_params
            s = bounds['S_MIN'] + s_norm * (bounds['S_MAX'] - bounds['S_MIN'])
            lb = bounds['LB_MIN'] + lb_norm * (bounds['LB_MAX'] - bounds['LB_MIN'])
            ub = bounds['UB_MIN'] + ub_norm * (bounds['UB_MAX'] - bounds['UB_MIN'])
            return s, lb, ub
        
        def objective_function(norm_params):
            try:
                s, lb, ub = denormalize_params(norm_params)
                
                # Check parameter validity
                if s <= 0 or ub <= lb:
                    return 1e6
                
                # Compute distribution values
                dist_values, _, _ = self.compute_func(s, lb, ub)
                
                # Calculate loss
                diff = np.mean(np.abs(dist_values - self.target_values) * self.weights)
                
                # Add regularization
                regularization = np.sum(norm_params)
                total_loss = diff + regularization
                
                if self.verbose and hasattr(self, '_opt_iteration'):
                    self._opt_iteration += 1
                    if self._opt_iteration % 50 == 0:
                        self.logger.debug(f"  Iteration {self._opt_iteration}: Loss={diff:.6f}, Total={total_loss:.6f}, "
                              f"S={s:.3f}, LB={lb:.6f}, UB={ub:.3f}")
                
                return total_loss
                
            except Exception as e:
                error_msg = f"Objective function failed: {str(e)}"
                self.logger.error(error_msg)
                self.optimization_errors.append({
                    'method': '_optimize_all_parameters.objective_function',
                    'error': error_msg,
                    'exception_type': type(e).__name__,
                    'parameters': norm_params.tolist() if hasattr(norm_params, 'tolist') else list(norm_params)
                })
                return 1e6
        
        # Set initial values
        s_init = 1
        lb_init = self.LB_init if self.LB_init is not None else bounds['LB_MIN'] * 10
        ub_init = self.UB_init if self.UB_init is not None else bounds['UB_MIN'] * 10
        
        # Ensure valid initial bounds
        if lb_init >= ub_init:
            lb_init = bounds['LB_MIN'] * 10
            ub_init = bounds['UB_MIN'] * 10
            
        initial_params = normalize_params(s_init, lb_init, ub_init)
        norm_bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        
        try:
            if self.verbose:
                self._opt_iteration = 0
                
            result = minimize(
                objective_function,
                initial_params,
                method=self.opt_method,
                bounds=norm_bounds,
                options={'maxiter': self.max_iterations, 'ftol': self.tolerance},
                tol=self.tolerance
            )
            
            s_opt, lb_opt, ub_opt = denormalize_params(result.x)
            
            # Validate results
            if lb_opt >= ub_opt or s_opt <= 0:
                self.logger.warning("Invalid optimized parameters, using fallback")
                return self._get_fallback_parameters()
            
            # Store optimization info
            self.optimization_results['all_params_optimization'] = {
                'success': result.success,
                'fun': float(result.fun),
                'nit': int(result.nit),
                'message': result.message
            }
            
            return s_opt, lb_opt, ub_opt
            
        except Exception as e:
            error_msg = f"All parameters optimization failed: {str(e)}"
            self.logger.error(error_msg)
            self.optimization_errors.append({
                'method': '_optimize_all_parameters',
                'error': error_msg,
                'exception_type': type(e).__name__
            })
                
            return self._get_fallback_parameters()

    def _optimize_s_parameter(self, lb: float, ub: float) -> float:
        """Optimize only S parameter with fixed bounds."""
        self.logger.info("Optimizing S parameter...")

        def objective_function(s_array):
            try:
                s = s_array[0]
                dist_values, _, _ = self.compute_func(s, lb, ub)
                diff = np.mean(np.abs(dist_values - self.target_values) * self.weights)

                self.logger.debug(f"  S optimization - Loss: {diff:.6f}, S: {s:.3f}")

                return diff
                
            except Exception as e:
                error_msg = f"S optimization objective failed: {str(e)}"
                self.logger.error(error_msg)
                self.optimization_errors.append({
                    'method': '_optimize_s_parameter.objective_function',
                    'error': error_msg,
                    'exception_type': type(e).__name__
                })
                return 1e6
        
        bounds = self._OPTIMIZATION_BOUNDS
        s_bounds = [(bounds['S_MIN'], bounds['S_MAX'])]
        
        try:
            result = minimize(
                objective_function,
                [1.0],  # Initial S value
                bounds=s_bounds,
                method=self.opt_method,
                options={'maxiter': 1000, 'ftol': self.tolerance}
            )
            
            # Store optimization info
            self.optimization_results['s_optimization'] = {
                'success': result.success,
                'fun': float(result.fun),
                'nit': int(result.nit),
                'message': result.message
            }
            
            return result.x[0]
            
        except Exception as e:
            error_msg = f"S parameter optimization failed: {str(e)}"
            self.logger.error(error_msg)
            self.optimization_errors.append({
                'method': '_optimize_s_parameter',
                'error': error_msg,
                'exception_type': type(e).__name__
            })
            
                
            return self._FALLBACK_VALUES['S']

    def _optimize_bounds_parameters(self, s: float) -> Tuple[float, float, float]:
        """Optimize LB and UB parameters with fixed S."""
        self.logger.info("Optimizing LB and UB parameters...")

        bounds = self._OPTIMIZATION_BOUNDS
        
        def normalize_bounds(lb, ub):
            lb_norm = (lb - bounds['LB_MIN']) / (bounds['LB_MAX'] - bounds['LB_MIN'])
            ub_norm = (ub - bounds['UB_MIN']) / (bounds['UB_MAX'] - bounds['UB_MIN'])
            return np.array([lb_norm, ub_norm])
        
        def denormalize_bounds(norm_params):
            lb_norm, ub_norm = norm_params
            lb = bounds['LB_MIN'] + lb_norm * (bounds['LB_MAX'] - bounds['LB_MIN'])
            ub = bounds['UB_MIN'] + ub_norm * (bounds['UB_MAX'] - bounds['UB_MIN'])
            return lb, ub
        
        def objective_function(norm_params):
            try:
                lb, ub = denormalize_bounds(norm_params)
                
                if lb <= 0 or ub <= lb:
                    return 1e6
                
                dist_values, _, _ = self.compute_func(s, lb, ub)
                diff = np.mean(np.abs(dist_values - self.target_values) * self.weights)
                
                # Add regularization
                regularization = np.sum(norm_params**2)
                total_loss = diff + regularization

                # print only 50th iteration
                if self.verbose and hasattr(self, '_opt_iteration'):
                    self._opt_iteration += 1
                    if self._opt_iteration % 50 == 0:
                        self.logger.debug(f"  Iteration {self._opt_iteration}: Loss={diff:.6f}, Total={total_loss:.6f}, "
                              f"LB={lb:.6f}, UB={ub:.3f}")

                return total_loss
                
            except Exception as e:
                error_msg = f"Bounds optimization objective failed: {str(e)}"
                self.optimization_errors.append({
                    'method': '_optimize_bounds_parameters.objective_function',
                    'error': error_msg,
                    'exception_type': type(e).__name__
                })
                return 1e6
        
        # Set initial values
        lb_init = self.LB_init if self.LB_init is not None else bounds['LB_MIN'] * 10
        ub_init = self.UB_init if self.UB_init is not None else bounds['UB_MIN'] * 10
        
        # Ensure valid bounds
        lb_init = np.clip(lb_init, bounds['LB_MIN'], bounds['LB_MAX'])
        ub_init = np.clip(ub_init, bounds['UB_MIN'], bounds['UB_MAX'])
        
        if lb_init >= ub_init:
            lb_init = bounds['LB_MIN'] * 10
            ub_init = bounds['UB_MIN'] * 10
        
        initial_params = normalize_bounds(lb_init, ub_init)
        norm_bounds = [(0.0, 1.0), (0.0, 1.0)]
        
        try:
            result = minimize(
                objective_function,
                initial_params,
                method=self.opt_method,
                bounds=norm_bounds,
                options={'maxiter': self.max_iterations, 'ftol': self.tolerance},
                tol=self.tolerance
            )
            
            lb_opt, ub_opt = denormalize_bounds(result.x)
            
            # Validate results
            if lb_opt >= ub_opt:
                if self.verbose:
                    self.logger.warning("Warning - Invalid optimized bounds, using initial values")
                return s, lb_init, ub_init
            
            # Store optimization info
            self.optimization_results['bounds_optimization'] = {
                'success': result.success,
                'fun': float(result.fun),
                'nit': int(result.nit),
                'message': result.message
            }
            
            return s, lb_opt, ub_opt
            
        except Exception as e:
            error_msg = f"Bounds optimization failed: {str(e)}"
            self.logger.error(error_msg)
            self.optimization_errors.append({
                'method': '_optimize_bounds_parameters',
                'error': error_msg,
                'exception_type': type(e).__name__
            })
            
            return s, lb_init, ub_init

    def _get_fallback_parameters(self) -> Tuple[float, float, float]:
        """Get fallback parameters when optimization fails."""
        self.logger.info("Using fallback parameters...")

        s_fallback = self._FALLBACK_VALUES['S']
        lb_fallback = self._estimate_fallback_lb()
        ub_fallback = self._estimate_fallback_ub()
        
        if self.verbose:
            self.logger.info(f"Using fallback parameters - S: {s_fallback}, "
                  f"LB: {lb_fallback}, UB: {ub_fallback}")
                  
        return s_fallback, lb_fallback, ub_fallback

    def _get_fallback_results(self) -> Dict[str, float]:
        """Get fallback results dictionary."""
        self.logger.info("Getting fallback results...")

        s_fallback, lb_fallback, ub_fallback = self._get_fallback_parameters()
        
        self.S_opt = s_fallback
        self.LB_opt = lb_fallback
        self.UB_opt = ub_fallback
        
        return {
            'S': float(s_fallback),
            'LB': float(lb_fallback),
            'UB': float(ub_fallback)
        }

    def _estimate_fallback_lb(self) -> float:
        """Estimate fallback LB value."""
        self.logger.info("Estimating fallback LB...")

        bounds = self._OPTIMIZATION_BOUNDS
        if self.LB_init is not None:
            return max(self.LB_init, bounds['LB_MIN'])
        else:
            return bounds['LB_MIN'] * self._FALLBACK_VALUES['LB_FACTOR']

    def _estimate_fallback_ub(self) -> float:
        """Estimate fallback UB value."""
        self.logger.info("Estimating fallback UB...")

        bounds = self._OPTIMIZATION_BOUNDS
        if self.UB_init is not None:
            return min(self.UB_init, bounds['UB_MAX'])
        else:
            return bounds['UB_MIN'] * self._FALLBACK_VALUES['UB_FACTOR']

    def get_optimization_info(self) -> Dict[str, Any]:
        """
        Get detailed optimization information.
        
        Returns:
        --------
        Dict[str, Any]
            Optimization results, errors, and warnings
        """
        self.logger.info("Getting optimization info...")

        return {
            'results': self.optimization_results.copy(),
            'errors': self.optimization_errors.copy(),
            'warnings': self.optimization_warnings.copy(),
            'optimized_parameters': {
                'S': self.S_opt,
                'LB': self.LB_opt,
                'UB': self.UB_opt
            } if self.S_opt is not None else None
        }

    def evaluate_with_optimized_parameters(self) -> Tuple[np.ndarray, Any, Any]:
        """
        Evaluate the compute function with optimized parameters.
        
        Returns:
        --------
        Tuple containing the results of compute_func(S_opt, LB_opt, UB_opt)
        """
        self.logger.info("Evaluating with optimized parameters...")
        if self.S_opt is None or self.LB_opt is None or self.UB_opt is None:
            self.logger.error("Parameters must be optimized before evaluation. Call optimize() first.")
            raise ValueError("Parameters must be optimized before evaluation. Call optimize() first.")
            
        return self.compute_func(self.S_opt, self.LB_opt, self.UB_opt)

    def compute_final_loss(self) -> float:
        """
        Compute final loss with optimized parameters.
        
        Returns:
        --------
        float
            Final weighted mean absolute error
        """
        self.logger.info("Computing final loss with optimized parameters...")
        try:
            dist_values, _, _ = self.evaluate_with_optimized_parameters()
            loss = np.mean(np.abs(dist_values - self.target_values) * self.weights)
            return float(loss)
        except Exception as e:
            error_msg = f"Final loss computation failed: {str(e)}"
            self.logger.error(error_msg)
            self.optimization_errors.append({
                'method': 'compute_final_loss',
                'error': error_msg,
                'exception_type': type(e).__name__
            })
            return float('inf')

    def reset(self):
        """Reset optimization state for reuse."""
        self.logger.info("Resetting engine state...")
        self.S_opt = None
        self.LB_opt = None
        self.UB_opt = None
        self.optimization_results.clear()
        self.optimization_errors.clear()
        self.optimization_warnings.clear()
        
        if self.verbose:
            self.logger.info("State reset successfully")

    def __repr__(self) -> str:
        """String representation of the engine."""
        self.logger.info("Getting string representation...")
        status = "optimized" if self.S_opt is not None else "not optimized"
        return f"DistFuncEngine(target_values={len(self.target_values)}, status={status})"