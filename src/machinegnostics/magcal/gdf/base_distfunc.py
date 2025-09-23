'''
Base Compute class for GDF

Author: Nirmal Parmar
Machine Gnostics
'''

import numpy as np
import warnings
from typing import Dict, Any, Tuple
from scipy.optimize import minimize
import logging
from machinegnostics.magcal.util.logging import get_logger
from machinegnostics.magcal.characteristics import GnosticsCharacteristics
from machinegnostics.magcal.gdf.base_df import BaseDistFunc
from machinegnostics.magcal.data_conversion import DataConversion
from machinegnostics.magcal.gdf.wedf import WEDF
from machinegnostics.magcal.mg_weights import GnosticsWeights
from machinegnostics.magcal.gdf.z0_estimator import Z0Estimator
from machinegnostics.magcal.gdf.distfunc_engine import DistFuncEngine

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
                 z0_optimize: bool = True,
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
        self.params = {}
        self.params['warnings'] = []
        self.params['errors'] = []
        self.data = data
        self.DLB = DLB
        self.DUB = DUB
        self.LB = LB
        self.UB = UB
        self.S = S
        self.z0_optimize = z0_optimize
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
        self._fitted = False
        self._derivatives_calculated = False
        self._marginal_analysis_done = False

        # safe for z0 compute
        self.pdf_points = None
        
        # Initialize computation cache
        self._computation_cache = {
            'data_converter': None,
            'characteristics_computer': None,
            'weights_normalized': None,
            'smooth_curves_generated': False
        }
        
        # log
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if verbose else logging.WARNING)
        self.logger.debug(f"{self.__class__.__name__} initialized:")
    # =============================================================================
    # VALIDATION AND INITIALIZATION
    # =============================================================================
    
    def _validate_inputs(self):
        """Comprehensive input validation with error and warning logging."""
        try:
            self.logger.info("Validating inputs.")
            # Data validation
            if not isinstance(self.data, np.ndarray):
                error_msg = "Data must be a numpy array."
                self.logger.error(error_msg)
                self.params['errors'].append({
                    'method': '_validate_inputs',
                    'error': error_msg,
                    'exception_type': 'TypeError',
                    'data_type_received': type(self.data).__name__
                })
                raise TypeError(error_msg)
                
            if self.data.size == 0:
                error_msg = "Data array cannot be empty."
                self.logger.error(error_msg)
                self.params['errors'].append({
                    'method': '_validate_inputs',
                    'error': error_msg,
                    'exception_type': 'ValueError',
                    'data_size': self.data.size
                })
                raise ValueError(error_msg)
                
            if not np.isfinite(self.data).all():
                non_finite_count = np.sum(~np.isfinite(self.data))
                error_msg = f"Data must contain only finite values. Found {non_finite_count} non-finite values."
                self.logger.error(error_msg)
                self.params['errors'].append({
                    'method': '_validate_inputs',
                    'error': error_msg,
                    'exception_type': 'ValueError',
                    'non_finite_count': int(non_finite_count),
                    'total_data_points': len(self.data)
                })
                raise ValueError(error_msg)
            
            # Data dimensional validation
            if self.data.ndim != 1:
                error_msg = f"Data must be a 1-dimensional array. Got {self.data.ndim}-dimensional array."
                self.logger.error(error_msg)
                self.params['errors'].append({
                    'method': '_validate_inputs',
                    'error': error_msg,
                    'exception_type': 'ValueError',
                    'data_shape': self.data.shape
                })
                raise ValueError(error_msg)
            
            # Bounds validation
            for bound, name in [(self.DLB, 'DLB'), (self.DUB, 'DUB'), (self.LB, 'LB'), (self.UB, 'UB')]:
                if bound is not None and (not isinstance(bound, (int, float)) or not np.isfinite(bound)):
                    error_msg = f"{name} must be a finite numeric value or None."
                    self.logger.error(error_msg)
                    self.params['errors'].append({
                        'method': '_validate_inputs',
                        'error': error_msg,
                        'exception_type': 'ValueError',
                        'parameter': name,
                        'value': bound,
                        'value_type': type(bound).__name__
                    })
                    raise ValueError(error_msg)
            
            # Bounds logical validation
            if self.DLB is not None and self.DUB is not None and self.DLB >= self.DUB:
                error_msg = f"DLB ({self.DLB}) must be less than DUB ({self.DUB}) when both are provided."
                self.logger.error(error_msg)
                self.params['errors'].append({
                    'method': '_validate_inputs',
                    'error': error_msg,
                    'exception_type': 'ValueError',
                    'DLB': float(self.DLB),
                    'DUB': float(self.DUB)
                })
                raise ValueError(error_msg)
                
            if self.LB is not None and self.UB is not None and self.LB >= self.UB:
                error_msg = f"LB ({self.LB}) must be less than UB ({self.UB}) when both are provided."
                self.logger.error(error_msg)
                self.params['errors'].append({
                    'method': '_validate_inputs',
                    'error': error_msg,
                    'exception_type': 'ValueError',
                    'LB': float(self.LB),
                    'UB': float(self.UB)
                })
                raise ValueError(error_msg)
            
            # S parameter validation
            if not isinstance(self.S, (int, float, str)):
                error_msg = "S must be a numeric positive value or 'auto'."
                self.logger.error(error_msg)
                self.params['errors'].append({
                    'method': '_validate_inputs',
                    'error': error_msg,
                    'exception_type': 'TypeError',
                    'S_type': type(self.S).__name__,
                    'S_value': self.S
                })
                raise TypeError(error_msg)
                
            if isinstance(self.S, (int, float)) and self.S <= 0:
                error_msg = f"S must be positive when specified as a number. Got {self.S}."
                self.logger.error(error_msg)
                self.params['errors'].append({
                    'method': '_validate_inputs',
                    'error': error_msg,
                    'exception_type': 'ValueError',
                    'S_value': float(self.S)
                })
                raise ValueError(error_msg)
                
            # S string validation when it's a string
            if isinstance(self.S, str) and self.S.lower() != 'auto':
                error_msg = f"When S is a string, it must be 'auto'. Got '{self.S}'."
                self.logger.error(error_msg)
                self.params['errors'].append({
                    'method': '_validate_inputs',
                    'error': error_msg,
                    'exception_type': 'ValueError',
                    'S_value': self.S
                })
                raise ValueError(error_msg)
            
            # varS parameter validation
            if not isinstance(self.varS, bool):
                error_msg = "varS must be a boolean value. VarS can be only true for 'ELDF' and 'QLDF'."
                self.logger.error(error_msg)
                self.params['errors'].append({
                    'method': '_validate_inputs',
                    'error': error_msg,
                    'exception_type': 'TypeError',
                    'varS_type': type(self.varS).__name__,
                    'varS_value': self.varS
                })
                raise TypeError(error_msg)
                
            # varS can be only true with S = 'auto'
            if self.varS and self.S != 'auto':
                error_msg = f"varS can only be true when S is set to 'auto'. Got S='{self.S}', varS={self.varS}."
                self.logger.error(error_msg)
                self.params['errors'].append({
                    'method': '_validate_inputs',
                    'error': error_msg,
                    'exception_type': 'ValueError',
                    'S_value': self.S,
                    'varS_value': self.varS
                })
                raise ValueError(error_msg)
        
            # Tolerance validation
            if not isinstance(self.tolerance, (int, float)) or self.tolerance <= 0:
                error_msg = f"Tolerance must be a positive numeric value. Got {self.tolerance}."
                self.logger.error(error_msg)
                self.params['errors'].append({
                    'method': '_validate_inputs',
                    'error': error_msg,
                    'exception_type': 'ValueError',
                    'tolerance_value': self.tolerance,
                    'tolerance_type': type(self.tolerance).__name__
                })
                raise ValueError(error_msg)
            
            # Tolerance range validation with warnings
            if self.tolerance > 1.0:
                warning_msg = f"tolerance ({self.tolerance}) is unusually large."
                self.logger.warning(warning_msg)
                self.params['warnings'].append({
                    'method': '_validate_inputs',
                    'message': warning_msg,
                    'severity': 'medium',
                    'tolerance_value': float(self.tolerance)
                })
                self.logger.info(f"Warning: {warning_msg}")

            if self.tolerance < 1e-10:
                warning_msg = f"tolerance ({self.tolerance}) is very small and may cause numerical issues."
                self.logger.warning(warning_msg)
                self.params['warnings'].append({
                    'method': '_validate_inputs',
                    'message': warning_msg,
                    'severity': 'high',
                    'tolerance_value': float(self.tolerance)
                })
                self.logger.info(f"Warning: {warning_msg}")

            if self.tolerance < 1e-10:
                warning_msg = f"tolerance ({self.tolerance}) is very small and may cause numerical issues."
                self.logger.warning(warning_msg)
                self.params['warnings'].append({
                    'method': '_validate_inputs',
                    'message': warning_msg,
                    'severity': 'high',
                    'tolerance_value': float(self.tolerance)
                })
                self.logger.info(f"Warning: {warning_msg}")

            # data_form validation
            if self.data_form not in ['a', 'm']:
                error_msg = f"data_form must be 'a' for additive or 'm' for multiplicative. Got '{self.data_form}'."
                self.logger.error(error_msg)
                self.params['errors'].append({
                    'method': '_validate_inputs',
                    'error': error_msg,
                    'exception_type': 'ValueError',
                    'data_form_value': self.data_form
                })
                raise ValueError(error_msg)
            
            # n_points validation
            if not isinstance(self.n_points, int) or self.n_points <= 0:
                error_msg = f"n_points must be a positive integer. Got {self.n_points}."
                self.logger.error(error_msg)
                self.params['errors'].append({
                    'method': '_validate_inputs',
                    'error': error_msg,
                    'exception_type': 'ValueError',
                    'n_points_value': self.n_points,
                    'n_points_type': type(self.n_points).__name__
                })
                raise ValueError(error_msg)
            
            # n_points reasonable range validation with warning
            if self.n_points > 10000:
                warning_msg = f"n_points ({self.n_points}) is very large and may impact performance."
                self.logger.warning(warning_msg)
                self.params['warnings'].append({
                    'method': '_validate_inputs',
                    'message': warning_msg,
                    'severity': 'medium',
                    'n_points_value': self.n_points
                })
                self.logger.warning(f"Warning: {warning_msg}")
            
            # Weights validation
            if self.weights is not None:
                if not isinstance(self.weights, np.ndarray):
                    error_msg = "weights must be a numpy array."
                    self.logger.error(error_msg)
                    self.params['errors'].append({
                        'method': '_validate_inputs',
                        'error': error_msg,
                        'exception_type': 'TypeError',
                        'weights_type': type(self.weights).__name__
                    })
                    raise TypeError(error_msg)
                    
                if len(self.weights) != len(self.data):
                    error_msg = f"Weights must have the same length as data. Got weights length {len(self.weights)}, data length {len(self.data)}."
                    self.logger.error(error_msg)
                    self.params['errors'].append({
                        'method': '_validate_inputs',
                        'error': error_msg,
                        'exception_type': 'ValueError',
                        'weights_length': len(self.weights),
                        'data_length': len(self.data)
                    })
                    raise ValueError(error_msg)
                    
                if not np.all(self.weights >= 0):
                    negative_count = np.sum(self.weights < 0)
                    error_msg = f"All weights must be non-negative. Found {negative_count} negative weights."
                    self.logger.error(error_msg)
                    self.params['errors'].append({
                        'method': '_validate_inputs',
                        'error': error_msg,
                        'exception_type': 'ValueError',
                        'negative_weights_count': int(negative_count)
                    })
                    raise ValueError(error_msg)
                    
                # Weights finite values validation
                if not np.isfinite(self.weights).all():
                    non_finite_weights = np.sum(~np.isfinite(self.weights))
                    error_msg = f"All weights must be finite values. Found {non_finite_weights} non-finite weights."
                    self.logger.error(error_msg)
                    self.params['errors'].append({
                        'method': '_validate_inputs',
                        'error': error_msg,
                        'exception_type': 'ValueError',
                        'non_finite_weights_count': int(non_finite_weights)
                    })
                    raise ValueError(error_msg)
            
            # z0_optimize validation
            if not isinstance(self.z0_optimize, bool):
                error_msg = f"z0_optimize must be a boolean value. Got {type(self.z0_optimize).__name__}."
                self.logger.error(error_msg)
                self.params['errors'].append({
                    'method': '_validate_inputs',
                    'error': error_msg,
                    'exception_type': 'TypeError',
                    'z0_optimize_type': type(self.z0_optimize).__name__,
                    'z0_optimize_value': self.z0_optimize
                })
                raise TypeError(error_msg)
            
            # opt_method validation
            valid_methods = ['L-BFGS-B', 'SLSQP', 'TNC', 'trust-constr', 'Powell', 'COBYLA']
            if not isinstance(self.opt_method, str):
                error_msg = f"opt_method must be a string. Got {type(self.opt_method).__name__}."
                self.logger.error(error_msg)
                self.params['errors'].append({
                    'method': '_validate_inputs',
                    'error': error_msg,
                    'exception_type': 'TypeError',
                    'opt_method_type': type(self.opt_method).__name__,
                    'opt_method_value': self.opt_method
                })
                raise TypeError(error_msg)
                
            if self.opt_method not in valid_methods:
                error_msg = f"opt_method must be one of {valid_methods}. Got '{self.opt_method}'."
                self.logger.error(error_msg)
                self.params['errors'].append({
                    'method': '_validate_inputs',
                    'error': error_msg,
                    'exception_type': 'ValueError',
                    'opt_method_value': self.opt_method,
                    'valid_methods': valid_methods
                })
                raise ValueError(error_msg)
            
            # max_data_size validation
            if not isinstance(self.max_data_size, int) or self.max_data_size <= 0:
                error_msg = f"max_data_size must be a positive integer. Got {self.max_data_size}."
                self.logger.error(error_msg)
                self.params['errors'].append({
                    'method': '_validate_inputs',
                    'error': error_msg,
                    'exception_type': 'ValueError',
                    'max_data_size_value': self.max_data_size,
                    'max_data_size_type': type(self.max_data_size).__name__
                })
                raise ValueError(error_msg)
            
            # flush parameter validation
            if not isinstance(self.flush, bool):
                error_msg = f"flush must be a boolean value. Got {type(self.flush).__name__}."
                self.logger.error(error_msg)
                self.params['errors'].append({
                    'method': '_validate_inputs',
                    'error': error_msg,
                    'exception_type': 'TypeError',
                    'flush_type': type(self.flush).__name__,
                    'flush_value': self.flush
                })
                raise TypeError(error_msg)
                
            # if length of data exceeds max_data_size, set flush to True with warning
            if len(self.data) > self.max_data_size and not self.flush:
                warning_msg = f"Data size ({len(self.data)}) exceeds max_data_size ({self.max_data_size}). For optimal compute performance, setting 'flush=True'."
                self.logger.warning(warning_msg)
                self.params['warnings'].append({
                    'method': '_validate_inputs',
                    'message': warning_msg,
                    'severity': 'medium',
                    'data_size': len(self.data),
                    'max_data_size': self.max_data_size,
                    'action_taken': 'flush_set_to_true'
                })
                self.flush = True
                self.logger.info(warning_msg)

            # Boolean parameters validation
            boolean_params = [
                (self.homogeneous, 'homogeneous'),
                (self.catch, 'catch'), 
                (self.wedf, 'wedf'),
                (self.verbose, 'verbose')
            ]
            
            for param, name in boolean_params:
                if not isinstance(param, bool):
                    error_msg = f"{name} must be a boolean value. Got {type(param).__name__}."
                    self.params['errors'].append({
                        'method': '_validate_inputs',
                        'error': error_msg,
                        'exception_type': 'TypeError',
                        'parameter': name,
                        'parameter_type': type(param).__name__,
                        'parameter_value': param
                    })
                    raise TypeError(error_msg)
        
        except Exception as e:
            if self.verbose:
                self.logger.error(f"Input validation failed: {str(e)}")
            raise
        
    def _store_initial_params(self):
        """Store initial parameters for reference."""
        self.logger.info("Storing initial parameters.")

        self.params.update({
            'data': np.sort(self.data).copy(),
            'DLB': self.DLB,
            'DUB': self.DUB,
            'LB': self.LB,
            'UB': self.UB,
            'S': self.S,
            'z0_optimize': self.z0_optimize,
            'varS': self.varS,
            'tolerance': self.tolerance,
            'data_form': self.data_form,
            'n_points': self.n_points,
            'homogeneous': self.homogeneous,
            'catch': self.catch,
            'weights': self.weights.copy() if self.weights is not None else None,
            'compute_wedf': self.wedf,
            'opt_method': self.opt_method,
            'verbose': self.verbose,
            'max_data_size': self.max_data_size,
            'flush': self.flush,
            'warnings': [],
            'errors': [],
            })

    # =============================================================================
    # DATA PREPROCESSING AND TRANSFORMATION
    # =============================================================================
    
    def _get_data_converter(self):
        """Get or create cached data converter."""
        self.logger.info("Retrieving data converter.")
        if self._computation_cache['data_converter'] is None:
            self._computation_cache['data_converter'] = DataConversion()
        return self._computation_cache['data_converter']

    def _estimate_data_bounds(self):
        """Estimate data bounds (DLB and DUB) if not provided."""
        self.logger.info("Estimating data bounds.")
        if self.DLB is None:
            self.DLB = np.min(self.data)
        if self.DUB is None:
            self.DUB = np.max(self.data)
        
        # Validate bounds
        if self.DLB >= self.DUB:
            self.logger.info("DLB >= DUB, All values are same case.")
        
        if self.catch:
            self.params.update({'DLB': float(self.DLB), 'DUB': float(self.DUB)})

    def _estimate_weights(self):
        """Process and normalize weights."""
        self.logger.info("Estimating and normalizing weights.")
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
            self.logger.info("Applying gnostic weights for non-homogeneous data.")
            gw = GnosticsWeights(verbose=self.verbose)
            self.gweights = gw._get_gnostic_weights(self.z)
            self.weights = self.gweights * self.weights
        
        # Cache normalized weights
        self._computation_cache['weights_normalized'] = self.weights.copy()
        
        if self.catch:
            self.params['weights'] = self.weights.copy()

    def _transform_data_to_standard_domain(self):
        """Transform data to standard z-domain."""
        self.logger.info("Transforming data to standard z-domain.")
        dc = self._get_data_converter()
        
        if self.data_form == 'a':
            self.z = dc._convert_az(self.data, self.DLB, self.DUB)
        elif self.data_form == 'm':
            self.z = dc._convert_mz(self.data, self.DLB, self.DUB)
        
        if self.catch:
            self.params['z'] = self.z.copy()

    def _generate_evaluation_points(self):
        """Generate points for smooth evaluation."""
        self.logger.info("Generating evaluation points.")
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
        self.logger.info("Estimating initial probable bounds (LB and UB).")
        
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
        self.logger.info("Computing distribution function values.")
        if use_wedf:
            self.logger.info("Using WEDF for distribution function computation.")
            wedf_ = WEDF(self.data, weights=self.weights, data_lb=self.DLB, data_ub=self.DUB, verbose=self.verbose)
            # if smooth:
            #     df_values = wedf_.fit(self.di_points_n)
            # else:
            df_values = wedf_.fit(self.data)
            
            if self.catch:
                self.params['wedf'] = df_values.copy()
            
            self.logger.info("WEDF values computed.")
            return df_values
        else:
            self.logger.info("Using KS points for distribution function computation.")
            # n_points = self.n_points if smooth else len(self.data)
            df_values = self._generate_ks_points(len(self.data))
            
            if self.catch:
                self.params['ksdf'] = df_values.copy()
            
            self.logger.info("KS points computed.")
            return df_values

    def _generate_ks_points(self, N):
        """Generate Kolmogorov-Smirnov points."""
        self.logger.info("Generating Kolmogorov-Smirnov points.")
        if N <= 0:
            raise ValueError("N must be a positive integer.")
        
        n = np.arange(1, N + 1)
        ks_points = (2 * n - 1) / (2 * N)
        
        if self.catch:
            self.params['ks_points'] = ks_points.copy()
        
        return ks_points

    def _determine_optimization_strategy(self, egdf: bool = True):
        """Determine optimization strategy for S, LB, and UB."""
        self.logger.info("Determining optimization strategy for S, LB, and UB.")
        try:
            self.logger.info("Initializing optimization Engine...")
                
            # For EGDF and QGDF optimization
            engine = DistFuncEngine(
                compute_func=self._compute_egdf_core if egdf else self._compute_qgdf_core, # NOTE switch between egdf and qgdf
                target_values=self.df_values,
                weights=self.weights,
                S=self.S,
                LB=self.LB,
                UB=self.UB,
                LB_init=self.LB_init,
                UB_init=self.UB_init,
                tolerance=self.tolerance,
                opt_method=self.opt_method,
                max_iterations=10000, # Engine will set default
                regularization_weight=None, # Engine will set default
                verbose=self.verbose,
                catch_errors=self.catch
            )

            results = engine.optimize()
            self.S_opt = results['S']
            self.LB_opt = results['LB']
            self.UB_opt = results['UB']

        except Exception as e:
            error_msg = f"Optimization strategy determination failed: {str(e)}"
            self.logger.error(error_msg)
            self.params['errors'].append({
                'method': '_determine_optimization_strategy',
                'error': error_msg,
                'exception_type': type(e).__name__
            })
            self.logger.error(error_msg)
            # Fallback to initial values
            self.logger.info("Falling back to initial values for S, LB, and UB.")
            self.S_opt = self.S if isinstance(self.S, (int, float)) else 1.0
            self.LB_opt = self.LB_init
            self.UB_opt = self.UB_init


    def _transform_bounds_to_original_domain(self):
        """Transform optimized bounds back to original domain."""
        dc = self._get_data_converter()

        self.logger.info("Transforming optimized bounds back to original domain.")
        
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

        self.logger.info("Cleaning up computation cache.")
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

        long_array_params = ['z_points', 'di_points', 'egdf_points', 'pdf_points', 'zi_n', 'zi_points', 'eldf_points', 'qldf_points', 'qgdf_points']

        for param in long_array_params:
            if param in self.params:
                self.params[param] = None
        
        if self.catch:
            self.params['computation_cache_cleared'] = True

        self.logger.info("Computation cache cleaned up.")


    def _calculate_fidelities_irrelevances_at_given_zi(self, zi):
        """Helper method to recalculate fidelities and irrelevances for current zi."""
        self.logger.info("Calculating fidelities and irrelevances at given zi.")

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

    
    def _calculate_gcq_at_given_zi(self, data) -> Tuple[GnosticsCharacteristics, np.ndarray, np.ndarray]:
        """Helper method to calculate q and q1 for current zi.
        this will be used in z0estimator for some methods and error calculation
        
        returns: gc, q, q1
        """
        self.logger.info("Calculating GnosticsCharacteristics, q, and q1 at given zi.")
        # conver to z domain with DLB and DUB
        zi = DataConversion._convert_az(data, self.DLB, self.DUB) if self.data_form == 'a' else DataConversion._convert_mz(data, self.DLB, self.DUB)
        # Convert to infinite domain
        zi_n = DataConversion._convert_fininf(self.z, self.LB_opt, self.UB_opt)
        # is data given then use it, else use self.zii
        zi_d = zi

        # Calculate R matrix
        eps = np.finfo(float).eps
        R = zi_n.reshape(-1, 1) / (zi_d + eps).reshape(1, -1)

        # Get characteristics
        gc = GnosticsCharacteristics(R=R, verbose=self.verbose)
        q, q1 = gc._get_q_q1(S=self.S_opt)

        return gc, q, q1

# NOTE: put this method to specific class that needs it, e.g., ELDF, QLDF, EGDF, QGDF
    # # z0 compute
    # def _compute_z0(self, optimize: bool = None):
    #     """
    #     Compute the Z0 point where PDF is maximum using the Z0Estimator class.
        
    #     Parameters:
    #     -----------
    #     optimize : bool, optional
    #         If True, use interpolation-based methods for higher accuracy.
    #         If False, use simple linear search on existing points.
    #         If None, uses the instance's z0_optimize setting.
    #     """
    #     if self.z is None:
    #         raise ValueError("Data must be transformed (self.z) before Z0 estimation.")
        
    #     # Use provided optimize parameter or fall back to instance setting
    #     use_optimize = optimize if optimize is not None else self.z0_optimize
        
    #     if self.verbose:
    #         print('GDF: Computing Z0 point using Z0Estimator...')

    #     try:
    #         # Create Z0Estimator instance with proper constructor signature
    #         z0_estimator = Z0Estimator(
    #             gdf_object=self,  # Pass the ELDF object itself
    #             optimize=use_optimize,
    #             verbose=self.verbose
    #         )
            
    #         # Call fit() method to estimate Z0
    #         self.z0 = z0_estimator.fit()
            
    #         # Get estimation info for debugging and storage
    #         if self.catch:
    #             estimation_info = z0_estimator.get_estimation_info()
    #             self.params.update({
    #                 'z0': float(self.z0),
    #                 'z0_method': estimation_info.get('z0_method', 'unknown'),
    #                 'z0_estimation_info': estimation_info
    #             })
            
    #         if self.verbose:
    #             method_used = z0_estimator.get_estimation_info().get('z0_method', 'unknown')
    #             print(f'ELDF: Z0 point computed successfully: {self.z0:.6f} (method: {method_used})')
                
    #     except Exception as e:
    #         # Log the error
    #         error_msg = f"Z0 estimation failed: {str(e)}"
    #         self.params['errors'].append({
    #             'method': '_compute_z0',
    #             'error': error_msg,
    #             'exception_type': type(e).__name__
    #         })

    #         if self.verbose:
    #             print(f"Warning: Z0Estimator failed with error: {e}")
    #             print("Falling back to simple maximum finding...")
            
    #         # Fallback to simple maximum finding
    #         self._compute_z0_fallback()
            
    #         if self.catch:
    #             self.params.update({
    #                 'z0': float(self.z0),
    #                 'z0_method': 'fallback_simple_maximum',
    #                 'z0_estimation_info': {'error': str(e)}
    #             })

    # def _compute_z0_fallback(self):
    #     """
    #     Fallback method for Z0 computation using simple maximum finding.
    #     """
    #     if not hasattr(self, 'di_points_n') or not hasattr(self, 'pdf_points'):
    #         raise ValueError("Both 'di_points_n' and 'pdf_points' must be defined for Z0 computation.")
        
    #     if self.verbose:
    #         print('Using fallback method for Z0 point...')
        
    #     # Find index with maximum PDF
    #     max_idx = np.argmax(self.pdf_points)
    #     self.z0 = self.di_points_n[max_idx]

    #     if self.verbose:
    #         print(f"Z0 point (fallback method): {self.z0:.6f}")

    # def analyze_z0(self, figsize: tuple = (12, 6)) -> Dict[str, Any]:
    #     """
    #     Analyze and visualize Z0 estimation results.
        
    #     Parameters:
    #     -----------
    #     figsize : tuple
    #         Figure size for the plot
            
    #     Returns:
    #     --------
    #     Dict[str, Any]
    #         Z0 analysis information
    #     """
    #     if not hasattr(self, 'z0') or self.z0 is None:
    #         raise ValueError("Z0 must be computed before analysis. Call fit() first.")
        
    #     # Create Z0Estimator for analysis
    #     z0_estimator = Z0Estimator(
    #         gdf_object=self,
    #         optimize=self.z0_optimize,
    #         verbose=self.verbose
    #     )
        
    #     # Re-estimate for analysis (this is safe since it's already computed)
    #     z0_estimator.fit()
        
    #     # Get detailed info
    #     analysis_info = z0_estimator.get_estimation_info()
        
    #     # Create visualization
    #     z0_estimator.plot_z0_analysis(figsize=figsize)
        
    #     return analysis_info
    
# NOTE: The following commented-out methods represent an earlier approach to optimization strategy determination. They have been replaced by the DistFuncEngine class for better modularity and maintainability.
    # def _determine_optimization_strategy(self):
    #     """Determine which parameters to optimize based on inputs."""
    #     if self.verbose:
    #         print("Determining optimization strategy...")
    #     s_is_auto = isinstance(self.S, str) and self.S.lower() == 'auto'
    #     lb_provided = self.LB is not None
    #     ub_provided = self.UB is not None
        
    #     if s_is_auto and not lb_provided and not ub_provided:
    #         # Optimize all parameters
    #         self.S_opt, self.LB_opt, self.UB_opt = self._optimize_all_parameters()
    #     elif lb_provided and ub_provided and s_is_auto:
    #         # Optimize only S
    #         self.LB_opt = self.LB_init
    #         self.UB_opt = self.UB_init
    #         self.S_opt = self._optimize_s_parameter(self.LB_opt, self.UB_opt)
    #     elif not s_is_auto and (not lb_provided or not ub_provided):
    #         # Optimize bounds only
    #         self.S_opt = self.S
    #         _, self.LB_opt, self.UB_opt = self._optimize_bounds_parameters(self.S_opt)
    #     else:
    #         # Use provided parameters
    #         self.S_opt = self.S if not s_is_auto else 1.0
    #         self.LB_opt = self.LB_init
    #         self.UB_opt = self.UB_init

    #     if self.verbose:
    #         print(f"Optimized parameters: S={self.S_opt:.6f}, LB={self.LB_opt:.6f}, UB={self.UB_opt:.6f}")

    # def _optimize_all_parameters(self):
    #     """Optimize all parameters using normalized parameter space."""
    #     if self.verbose:
    #         print("Optimizing all parameters (S, LB, UB)...")
    #     bounds = self._OPTIMIZATION_BOUNDS
        
    #     def normalize_params(s, lb, ub):
    #         s_norm = (s - bounds['S_MIN']) / (bounds['S_MAX'] - bounds['S_MIN'])
    #         lb_norm = (lb - bounds['LB_MIN']) / (bounds['LB_MAX'] - bounds['LB_MIN'])
    #         ub_norm = (ub - bounds['UB_MIN']) / (bounds['UB_MAX'] - bounds['UB_MIN'])
    #         return s_norm, lb_norm, ub_norm
        
    #     def denormalize_params(s_norm, lb_norm, ub_norm):
    #         s = bounds['S_MIN'] + s_norm * (bounds['S_MAX'] - bounds['S_MIN'])
    #         lb = bounds['LB_MIN'] + lb_norm * (bounds['LB_MAX'] - bounds['LB_MIN'])
    #         ub = bounds['UB_MIN'] + ub_norm * (bounds['UB_MAX'] - bounds['UB_MIN'])
    #         return s, lb, ub
        
    #     def objective_function(norm_params):
    #         try:
    #             s, lb, ub = denormalize_params(*norm_params)
                
    #             if s <= 0 or ub <= lb:
    #                 return 1e6
                
    #             egdf_values, _, _ = self._compute_egdf_core(s, lb, ub)
    #             diff = np.mean(np.abs(egdf_values - self.df_values) * self.weights)
                
    #             # Regularization
    #             reg = np.sum(np.array(norm_params)**2)
                
    #             total_loss = diff + reg
                
    #             if self.verbose:
    #                 print(f"Loss: {diff:.6f}, Total: {total_loss:.6f}, S: {s:.3f}, LB: {lb:.6f}, UB: {ub:.3f}")
                
    #             return total_loss
    #         except:
    #             error_msg = f"Objective function computation failed: {str(e)}"
    #             self.params['errors'].append({
    #                 'method': '_optimize_all_parameters.objective_function',
    #                 'error': error_msg,
    #                 'exception_type': type(e).__name__,
    #                 'norm_params': norm_params.tolist() if hasattr(norm_params, 'tolist') else list(norm_params)
    #             })
    #             return 1e6
        
    #     # Initial values
    #     s_init = 0.05
    #     lb_init = self.LB_init if hasattr(self, 'LB_init') and self.LB_init is not None else bounds['LB_MIN']
    #     ub_init = self.UB_init if hasattr(self, 'UB_init') and self.UB_init is not None else bounds['UB_MAX']
        
    #     initial_params = normalize_params(s_init, lb_init, ub_init)
    #     norm_bounds = [(0.0, 1.0)]
        
    #     try:
    #         result = minimize(
    #             objective_function,
    #             initial_params,
    #             method=self.opt_method,
    #             bounds=norm_bounds,
    #             options={'maxiter': 10000, 'ftol': self.tolerance},
    #             tol=self.tolerance  
    #         )
            
    #         s_opt, lb_opt, ub_opt = denormalize_params(*result.x)
            
    #         if lb_opt >= ub_opt:
    #             if self.verbose:
    #                 print("Warning: Optimized LB >= UB, using initial values")
    #             return s_init, lb_init, ub_init
            
    #         return s_opt, lb_opt, ub_opt
    #     except Exception as e:
    #         # error handling
    #         error_msg = f"Optimization failed: {str(e)}"
    #         self.params['errors'].append({
    #             'method': '_optimize_all_parameters',
    #             'error': error_msg,
    #             'exception_type': type(e).__name__
    #         })
    #         if self.verbose:
    #             print(f"Optimization failed: {e}")
    #         return s_init, lb_init, ub_init

    # def _optimize_s_parameter(self, lb, ub):
    #     """Optimize only S parameter."""
    #     if self.verbose:
    #         print("Optimizing S parameter...")

    #     def objective_function(s):
    #         try:
    #             egdf_values, _, _ = self._compute_egdf_core(s[0], lb, ub)
    #             diff = np.mean(np.abs(egdf_values - self.df_values) * self.weights)
    #             if self.verbose:
    #                 print(f"S optimization - Loss: {diff:.6f}, S: {s[0]:.3f}")
    #             return diff
    #         except Exception as e:
    #             error_msg = f"S optimization objective function failed: {str(e)}"
    #             self.params['errors'].append({
    #                 'method': '_optimize_s_parameter',
    #                 'error': error_msg,
    #                 'exception_type': type(e).__name__
    #             })
    #             return 1e6
        
    #     try:
    #         result = minimize(
    #             objective_function,
    #             [1.0],
    #             bounds=[(self._OPTIMIZATION_BOUNDS['S_MIN'], self._OPTIMIZATION_BOUNDS['S_MAX'])],
    #             method=self.opt_method,
    #             options={'maxiter': 1000}
    #         )
    #         return result.x[0]
    #     except Exception as e:
    #         error_msg = f"S optimization failed: {str(e)}"
    #         self.params['errors'].append({
    #             'method': '_optimize_s_parameter',
    #             'error': error_msg,
    #             'exception_type': type(e).__name__
    #         })
    #         return 1.0

    # def _optimize_bounds_parameters(self, s):
    #     """Optimize only LB and UB parameters."""
    #     if self.verbose:
    #         print("Optimizing LB and UB parameters...")
            
    #     bounds = self._OPTIMIZATION_BOUNDS
        
    #     def normalize_bounds(lb, ub):
    #         lb_norm = (lb - bounds['LB_MIN']) / (bounds['LB_MAX'] - bounds['LB_MIN'])
    #         ub_norm = (ub - bounds['UB_MIN']) / (bounds['UB_MAX'] - bounds['UB_MIN'])
    #         return lb_norm, ub_norm
        
    #     def denormalize_bounds(lb_norm, ub_norm):
    #         lb = bounds['LB_MIN'] + lb_norm * (bounds['LB_MAX'] - bounds['LB_MIN'])
    #         ub = bounds['UB_MIN'] + ub_norm * (bounds['UB_MAX'] - bounds['UB_MIN'])
    #         return lb, ub
        
    #     def objective_function(norm_params):
    #         try:
    #             lb, ub = denormalize_bounds(*norm_params)
                
    #             if lb <= 0 or ub <= lb:
    #                 return 1e6
                
    #             egdf_values, _, _ = self._compute_egdf_core(s, lb, ub)
    #             diff = np.mean(np.abs(egdf_values - self.df_values) * self.weights)
                
    #             # Regularization
    #             reg = np.sum(np.array(norm_params)**2)
    #             total_loss = diff + reg
                
    #             if self.verbose:
    #                 print(f"Bounds optimization - Loss: {diff:.6f}, Total: {total_loss:.6f}, LB: {lb:.6f}, UB: {ub:.3f}")
    #         except Exception as e:
    #             error_msg = f"Bounds optimization objective function failed: {str(e)}"
    #             self.params['errors'].append({
    #                 'method': '_optimize_bounds_parameters',
    #                 'error': error_msg,
    #                 'exception_type': type(e).__name__
    #             })
    #             return 1e6
        
    #     # Initial values
    #     lb_init = self.LB_init if hasattr(self, 'LB_init') and self.LB_init is not None else bounds['LB_MIN']
    #     ub_init = self.UB_init if hasattr(self, 'UB_init') and self.UB_init is not None else bounds['UB_MIN']
        
    #     lb_init = np.clip(lb_init, bounds['LB_MIN'], bounds['LB_MAX'])
    #     ub_init = np.clip(ub_init, bounds['UB_MIN'], bounds['UB_MAX'])
        
    #     if lb_init >= ub_init:
    #         lb_init = bounds['LB_MIN']
    #         ub_init = bounds['UB_MIN']
        
    #     initial_params = normalize_bounds(lb_init, ub_init)
    #     norm_bounds = [(0.0, 1.0), (0.0, 1.0)]
        
    #     try:
    #         result = minimize(
    #             objective_function,
    #             initial_params,
    #             method=self.opt_method,
    #             bounds=norm_bounds,
    #             options={'maxiter': 10000, 'ftol': self.tolerance},
    #             tol=self.tolerance
    #         )
            
    #         lb_opt, ub_opt = denormalize_bounds(*result.x)
            
    #         if lb_opt >= ub_opt:
    #             if self.verbose:
    #                 print("Warning: Optimized LB >= UB, using initial values")
    #             return s, lb_init, ub_init
            
    #         return s, lb_opt, ub_opt
    #     except Exception as e:
    #         error_msg = f"Bounds optimization failed: {str(e)}"
    #         self.params['errors'].append({
    #             'method': '_optimize_bounds_parameters',
    #             'error': error_msg,
    #             'exception_type': type(e).__name__
    #         })
    #         if self.verbose:
    #             print(f"Bounds optimization failed: {e}")
    #         return s, self.LB, self.UB
