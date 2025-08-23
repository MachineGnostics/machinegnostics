'''
Base Interval Analysis for EGDF

**Interval Analysis - Critical Points:**
    
    5. **Z0**: Central point where PDF reaches global maximum and EGDF ≈ 0.5. This represents the
       distribution's central tendency and inflection point.
    
    6. **Z0L and Z0U**: Lower and upper bounds of the tolerance interval around Z0.
    
    7. **ZL and ZU**: Bounds defining the interval of typical data values.
'''

import numpy as np
import warnings
from machinegnostics.magcal.gdf.egdf import EGDF
from machinegnostics.magcal.gdf.base_eg_ma import BaseMarginalAnalysisEGDF
from machinegnostics.magcal.gdf.homogeneity import DataHomogeneity

class BaseIntervalAnalysisEGDF(BaseMarginalAnalysisEGDF):
    """
    Base class for interval analysis in EGDF.
    
    Attributes:
        DLB (float): Lower bound of the data range.
        DUB (float): Upper bound of the data range.
        LSB (float): Lower bound of the sample.
        USB (float): Upper bound of the sample.
        z0 (float): Central point where PDF reaches global maximum and EGDF ≈ 0.5.
    """
    
    def __init__(self,
                data: np.ndarray,
                estimate_sample_bounds: bool = False,
                estimate_cluster_bounds: bool = False,
                sample_bound_tolerance: float = 0.1,
                max_iterations: int = 1000, # NOTE for intv specific
                early_stopping_steps: int = 10,
                estimating_rate: float = 0.01, # NOTE for intv specific
                cluster_threshold: float = 0.05,
                linear_search: bool = True, # NOTE for intv specific
                get_clusters: bool = False, # NOTE for intv specific
                DLB: float = None,
                DUB: float = None,
                LB: float = None,
                UB: float = None,
                S = 'auto',
                tolerance: float = 1e-9, # NOTE for intv specific
                data_form: str = 'a',
                n_points: int = 1000, # NOTE for intv specific
                homogeneous: bool = True,
                catch: bool = True,
                weights: np.ndarray = None,
                wedf: bool = True,
                opt_method: str = 'L-BFGS-B',
                verbose: bool = False,
                max_data_size: int = 1000,
                flush: bool = True): # NOTE for intv specific
        super().__init__(data=data, 
                         sample_bound_tolerance=sample_bound_tolerance, 
                         max_iterations=max_iterations, 
                         early_stopping_steps=early_stopping_steps,
                         estimating_rate=estimating_rate, 
                         cluster_threshold=cluster_threshold, 
                         get_clusters=get_clusters, 
                         DLB=DLB, 
                         DUB=DUB, 
                         LB=LB, 
                         UB=UB,
                         S=S, 
                         tolerance=tolerance, 
                         data_form=data_form, 
                         n_points=n_points, 
                         homogeneous=homogeneous, 
                         catch=catch, 
                         weights=weights,
                         wedf=wedf, 
                         opt_method=opt_method, 
                         verbose=verbose, 
                         max_data_size=max_data_size, 
                         flush=flush)
        # Initialize interval-specific attributes
        self.Z0 = None
        self.Z0L = None
        self.Z0U = None
        self.ZL = None
        self.ZU = None
        self.estimate_sample_bounds = estimate_sample_bounds
        self.estimate_cluster_bounds = estimate_cluster_bounds
        self.linear_search = linear_search
        self.params = {}

        # input validation
        if not isinstance(self.estimate_sample_bounds, bool):
            raise ValueError("estimate_sample_bounds must be a boolean.")
        if not isinstance(self.estimate_cluster_bounds, bool):
            raise ValueError("estimate_cluster_bounds must be a boolean.")
        if not isinstance(self.linear_search, bool):
            raise ValueError("linear_search must be a boolean.")
        # if get_clusters is True, then estimate_cluster_bounds must be True
        if self.get_clusters and not self.estimate_cluster_bounds:
            raise ValueError(
                "Invalid parameter combination: 'get_clusters=True' requires 'estimate_cluster_bounds=True'. "
                "To get cluster information, cluster bounds must first be estimated. "
                "Please set 'estimate_cluster_bounds=True' or set 'get_clusters=False'."
                )

        # max data size validation
        if len(self.data) > self.max_data_size:
            warnings.warn(f"Data length ({len(self.data)}) exceeds max_data_size ({self.max_data_size}). Set max_data_size to a larger value if hardware allows. OR set `flush=True` and `catch=False` to clear data after each iteration.")

    def _create_extended_egdf_intv(self, datum):
        """Create EGDF with extended data including the given datum."""
        if self.get_clusters:
            data = self.init_egdf.params['main_cluster']
            # is main cluster empty? then fall back to init data
            if data is None or len(data) == 0:
                warnings.warn("Main cluster is empty, using initial EGDF data instead.")
                data = self.init_egdf.data
        else:
            data = self.init_egdf.data

        data_extended = np.append(data, datum)

        egdf_extended = EGDF(
            data=data_extended,
            tolerance=self.tolerance,
            data_form=self.data_form,
            n_points=self.n_points,
            homogeneous=self.homogeneous,
            S=self.S_opt,
            LB=self.LB,
            UB=self.UB,
            catch=False,
            weights=self.weights,
            wedf=self.wedf,
            opt_method=self.opt_method,
            verbose=False,
            max_data_size=self.max_data_size,
            flush=False
        )
        
        egdf_extended.fit()
        return egdf_extended
    
    def _compute_intv_scipy(self):
        '''
        using scipy minimize with constraints to find Z0L and Z0U with improved robustness
        '''
        if self.verbose:
            print("Computing interval values using scipy optimization with constraints...")
    
        try:
            from scipy.optimize import minimize
            import warnings
        except ImportError as e:
            raise ImportError("scipy is required for this method. Please install scipy.") from e
        
        # For ZL optimization: minimize Z0 subject to LB <= zl <= z0_main
        zol_bounds = [(self.LB, self._z0_main)]
        
        # For ZU optimization: maximize Z0 subject to z0_main <= zu <= UB  
        zou_bounds = [(self._z0_main, self.UB)]
    
        # Improved constraint functions with tolerance
        def constraint_z0l_value(x):
            """Ensure resulting Z0L <= z0_main"""
            try:
                z_egdf = self._create_extended_egdf_intv(x[0])
                z0_datum = self._get_z0(z_egdf)
                return self._z0_main - z0_datum + self.tolerance  # Add tolerance for numerical stability
            except Exception:
                return -1e6  # Large negative value if computation fails
    
        def constraint_z0u_value(x):
            """Ensure resulting Z0U >= z0_main"""
            try:
                z_egdf = self._create_extended_egdf_intv(x[0])
                z0_datum = self._get_z0(z_egdf)
                return z0_datum - self._z0_main + self.tolerance  # Add tolerance for numerical stability
            except Exception:
                return -1e6  # Large negative value if computation fails
    
        # Define constraints for scipy
        constraints_zol = [
            {'type': 'ineq', 'fun': constraint_z0l_value}
        ]
        
        constraints_zou = [
            {'type': 'ineq', 'fun': constraint_z0u_value}
        ]
    
        # Robust objective functions with error handling
        def objective_zol(datum):
            try:
                z_egdf = self._create_extended_egdf_intv(datum[0])
                z0_datum = self._get_z0(z_egdf)
                return z0_datum
            except Exception:
                return 1e6  # Large positive value if computation fails
        
        def objective_zou(datum):
            try:
                z_egdf = self._create_extended_egdf_intv(datum[0])
                z0_datum = self._get_z0(z_egdf)
                return -z0_datum  # negative because we want to maximize Z0
            except Exception:
                return 1e6  # Large positive value if computation fails
    
        # Multiple optimization attempts with different initial points and methods
        methods = ['SLSQP', 'trust-constr']
        
        # Initialize with fallback values
        self.zl = float(self._z0_main)
        self.z0l = float(self._z0_main)
        self.zu = float(self._z0_main)
        self.z0u = float(self._z0_main)
        
        zol_success = False
        zou_success = False
        
        for method in methods:
            if zol_success and zou_success:
                break
                
            # Try different initial points for ZL optimization
            if not zol_success:
                initial_points_zol = [
                    [self._z0_main],
                    [self.LB + 0.1 * (self._z0_main - self.LB)],
                    [self.LB + 0.5 * (self._z0_main - self.LB)],
                    [self.LB + 0.9 * (self._z0_main - self.LB)]
                ]
                
                for x0_zol in initial_points_zol:
                    try:
                        res_zol = minimize(
                            objective_zol, 
                            x0=np.array(x0_zol), 
                            method=method,
                            bounds=zol_bounds,
                            constraints=constraints_zol,
                            options={'ftol': max(self.tolerance, 1e-12), 'maxiter': 1000}
                        )
                        
                        if res_zol.success and res_zol.fun <= self._z0_main + self.tolerance:
                            self.zl = float(res_zol.x[0])
                            self.z0l = float(res_zol.fun)
                            zol_success = True
                            break
                    except Exception:
                        continue
            
            # Try different initial points for ZU optimization
            if not zou_success:
                initial_points_zou = [
                    [self._z0_main],
                    [self._z0_main + 0.1 * (self.UB - self._z0_main)],
                    [self._z0_main + 0.5 * (self.UB - self._z0_main)],
                    [self._z0_main + 0.9 * (self.UB - self._z0_main)]
                ]
                
                for x0_zou in initial_points_zou:
                    try:
                        res_zou = minimize(
                            objective_zou,
                            x0=np.array(x0_zou), 
                            method=method,
                            bounds=zou_bounds,
                            constraints=constraints_zou,
                            options={'ftol': max(self.tolerance, 1e-12), 'maxiter': 1000}
                        )
                        
                        if res_zou.success and (-res_zou.fun) >= self._z0_main - self.tolerance:
                            self.zu = float(res_zou.x[0])
                            self.z0u = float(-res_zou.fun)
                            zou_success = True
                            break
                    except Exception:
                        continue
    
        # Post-processing to ensure ordering
        if self.z0l > self._z0_main:
            self.z0l = float(self._z0_main)
            zol_success = False
        
        if self.z0u < self._z0_main:
            self.z0u = float(self._z0_main)
            zou_success = False
        
        # Ensure datum ordering by adjusting if necessary
        if self.zl > self._z0_main:
            self.zl = float(self._z0_main)
        
        if self.zu < self._z0_main:
            self.zu = float(self._z0_main)
        
        # Set Z0
        self.z0 = float(self._z0_main)
    
        # Final validation and correction
        ordering_satisfied = (self.zl <= self.z0l <= self.z0 <= self.z0u <= self.zu)
        
        if not ordering_satisfied:
            warnings.warn("Ordering constraint violated. Applying corrections...")
            
            # Apply minimum corrections to satisfy ordering
            if self.zl > self.z0l:
                self.zl = self.z0l
            if self.z0l > self.z0:
                self.z0l = self.z0
            if self.z0 > self.z0u:
                self.z0u = self.z0
            if self.z0u > self.zu:
                self.zu = self.z0u
            
            ordering_satisfied = (self.zl <= self.z0l <= self.z0 <= self.z0u <= self.zu)
        
        if self.verbose:
            print(f"\nInterval Analysis Results (Scipy Optimization with Constraints):")
            print(f"ZL:  {self.zl:.6f} (datum producing minimum Z0)")
            print(f"Z0L: {self.z0l:.6f} (minimum Z0 value)")
            print(f"Z0:  {self.z0:.6f} (original central point)")
            print(f"Z0U: {self.z0u:.6f} (maximum Z0 value)")
            print(f"ZU:  {self.zu:.6f} (datum producing maximum Z0)")
            print(f"Ordering constraint satisfied: {ordering_satisfied}")
            print(f"ZL optimization: {'Success' if zol_success else 'Failed'}")
            print(f"ZU optimization: {'Success' if zou_success else 'Failed'}")
    
        if self.catch:
            self.params = getattr(self.init_egdf, 'params', {}).copy()
            self.params.update({
                'ZL': self.zl,
                'Z0L': self.z0l,
                'Z0': float(self.z0),
                'Z0U': self.z0u,
                'ZU': self.zu,
                'optimization_success': zol_success and zou_success,
                'ordering_satisfied': ordering_satisfied
            })
    
    def _compute_intv(self):
        '''
        first compute interval using scipy minimize
        if it fails then use linear search method
        '''
        if self.verbose:
            print("Initiating interval computation...")

        # compute with fallback
        try:
            self._compute_intv_scipy()
        except Exception as e:
            warnings.warn(f"Scipy optimization failed: {e}. Falling back to linear search method.")
            self._compute_intv_linear_search()


    def _compute_intv_linear_search(self): # NOTE in future, this computation may be optimized
        """
        Compute interval values including Z0L, Z0U, ZL, and ZU.
        
        This method calculates critical interval boundaries based on the Z0 interval
        and the central point Z0. It identifies tolerance bounds and typical value ranges.
        Enhanced with convergence detection for plateau behavior and improved logic.
        """
        try:
            # Input validation
            if not hasattr(self, 'z0') or self.z0 is None:
                raise ValueError("Z0 must be computed before interval analysis. Call _get_z0() first.")
            
            if not hasattr(self, 'LB') or not hasattr(self, 'UB') or self.LB is None or self.UB is None:
                raise ValueError("LB and UB must be defined for interval computation.")
            
            if self.LB >= self.UB:
                raise ValueError(f"Lower bound ({self.LB}) must be less than upper bound ({self.UB}).")
            
            if not (self.LB <= self.z0 <= self.UB):
                warnings.warn(f"Z0 ({self.z0}) is outside the bounds [{self.LB}, {self.UB}].")
            
            # Initialize empty lists and variables
            self.z0_interval = []
            self.datum_range = []
            
            # Initialize interval tracking with separate lists for lower and upper searches
            lower_search_data = {'datum': [], 'z0': []}
            upper_search_data = {'datum': [], 'z0': []}
            
            if self.verbose:
                print("Computing interval values...")
                print(f"Z0: {self.z0:.6f}, LB: {self.LB:.6f}, UB: {self.UB:.6f}")
            
            # Ensure we have enough points for meaningful analysis
            min_points_per_side = max(100, self.n_points)
            points_per_side = max(min_points_per_side, self.n_points)
            
            # Initialize tracking variables
            current_z0_min = self.z0
            current_z0_max = self.z0
            z0l_datum = self.z0
            z0u_datum = self.z0
            
            # Early stopping parameters
            early_stop_tolerance = self._TOLERANCE
            consecutive_increases = 0
            consecutive_decreases = 0
            max_consecutive = 5
            
            # Convergence/Plateau detection parameters
            convergence_tolerance = self._TOLERANCE
            plateau_window = 5
            min_plateau_points = 5
            
            # Search towards lower bound
            if self._z0_main > self.LB:
                lower_range = np.linspace(self._z0_main, self.LB, points_per_side)
                z0_history_lower = []
                lower_z0_min = self._z0_main
                lower_z0l_datum = self._z0_main

                for i, datum in enumerate(lower_range[1:], 1):
                    try:
                        if self.verbose and i % max(1, points_per_side // 10) == 0:
                            print(f"Processing lower range: {datum:.6f} ({i}/{len(lower_range)-1})")
                        
                        z_egdf = self._create_extended_egdf_intv(datum)
                        z0_datum = self._get_z0(z_egdf)
                        
                        # Store in separate lower search data
                        lower_search_data['datum'].append(datum)
                        lower_search_data['z0'].append(z0_datum)
                        
                        # Also store in combined arrays for backward compatibility
                        self.z0_interval.append(z0_datum)
                        self.datum_range.append(datum)
                        z0_history_lower.append(z0_datum)
                        
                        # Track minimum Z0 in lower search specifically
                        if z0_datum < lower_z0_min:
                            lower_z0_min = z0_datum
                            lower_z0l_datum = datum
                            consecutive_increases = 0
                        else:
                            consecutive_increases += 1
                        
                        # Update global minimum
                        if z0_datum < current_z0_min:
                            current_z0_min = z0_datum
                        
                        # Convergence/Plateau detection (same as before)
                        if len(z0_history_lower) >= min_plateau_points:
                            recent_window = min(plateau_window, len(z0_history_lower))
                            recent_z0_values = z0_history_lower[-recent_window:]
                            
                            z0_variance = np.var(recent_z0_values)
                            z0_range = np.max(recent_z0_values) - np.min(recent_z0_values)
                            
                            if (z0_variance < convergence_tolerance and z0_range < convergence_tolerance):
                                if self.verbose:
                                    print(f"Convergence detected at lower bound: Z0 plateau reached")
                                break
                            
                            mean_recent_z0 = np.mean(recent_z0_values)
                            max_deviation = np.max(np.abs(recent_z0_values - mean_recent_z0))
                            
                            if max_deviation < convergence_tolerance:
                                if self.verbose:
                                    print(f"Convergence detected at lower bound: Z0 stabilized")
                                break
                        
                        # Early stopping logic
                        if (z0_datum > lower_z0_min + early_stop_tolerance and 
                            consecutive_increases >= max_consecutive):
                            if self.verbose:
                                print(f"Early stopping at lower bound: Z0 increasing for {consecutive_increases} consecutive points")
                            break
                            
                    except Exception as e:
                        warnings.warn(f"Error processing datum {datum:.6f} in lower range: {e}")
                        continue
            
            # Search towards upper bound
            if self._z0_main < self.UB:
                upper_range = np.linspace(self._z0_main, self.UB, points_per_side)
                z0_history_upper = []
                upper_z0_max = self._z0_main
                upper_z0u_datum = self._z0_main

                for i, datum in enumerate(upper_range[1:], 1):
                    try:
                        if self.verbose and i % max(1, points_per_side // 10) == 0:
                            print(f"Processing upper range: {datum:.6f} ({i}/{len(upper_range)-1})")
                        
                        z_egdf = self._create_extended_egdf_intv(datum)
                        z0_datum = self._get_z0(z_egdf)
                        
                        # Store in separate upper search data
                        upper_search_data['datum'].append(datum)
                        upper_search_data['z0'].append(z0_datum)
                        
                        # Also store in combined arrays for backward compatibility
                        self.z0_interval.append(z0_datum)
                        self.datum_range.append(datum)
                        z0_history_upper.append(z0_datum)
                        
                        # Track maximum Z0 in upper search specifically
                        if z0_datum > upper_z0_max:
                            upper_z0_max = z0_datum
                            upper_z0u_datum = datum
                            consecutive_decreases = 0
                        else:
                            consecutive_decreases += 1
                        
                        # Update global maximum
                        if z0_datum > current_z0_max:
                            current_z0_max = z0_datum
                        
                        # Convergence/Plateau detection (same as before)
                        if len(z0_history_upper) >= min_plateau_points:
                            recent_window = min(plateau_window, len(z0_history_upper))
                            recent_z0_values = z0_history_upper[-recent_window:]
                            
                            z0_variance = np.var(recent_z0_values)
                            z0_range = np.max(recent_z0_values) - np.min(recent_z0_values)
                            
                            if (z0_variance < convergence_tolerance and z0_range < convergence_tolerance):
                                if self.verbose:
                                    print(f"Convergence detected at upper bound: Z0 plateau reached")
                                break
                            
                            mean_recent_z0 = np.mean(recent_z0_values)
                            max_deviation = np.max(np.abs(recent_z0_values - mean_recent_z0))
                            
                            if max_deviation < convergence_tolerance:
                                if self.verbose:
                                    print(f"Convergence detected at upper bound: Z0 stabilized")
                                break
                        
                        # Early stopping logic
                        if (z0_datum < upper_z0_max - early_stop_tolerance and 
                            consecutive_decreases >= max_consecutive):
                            if self.verbose:
                                print(f"Early stopping at upper bound: Z0 decreasing for {consecutive_decreases} consecutive points")
                            break
                            
                    except Exception as e:
                        warnings.warn(f"Error processing datum {datum:.6f} in upper range: {e}")
                        continue
            
            # Add Z0 point if not already included
            if self.z0 not in self.datum_range:
                self.z0_interval.append(self.z0)
                self.datum_range.append(self.z0)
            
            # Validate we have enough data points
            if len(self.z0_interval) < 3:
                warnings.warn("Insufficient data points for reliable interval analysis.")
            
            # Convert to numpy arrays
            self.z0_interval = np.array(self.z0_interval)
            self.datum_range = np.array(self.datum_range)
            
            # Remove any NaN or infinite values
            valid_mask = np.isfinite(self.z0_interval) & np.isfinite(self.datum_range)
            if not np.all(valid_mask):
                warnings.warn(f"Removing {np.sum(~valid_mask)} invalid data points from interval analysis.")
                self.z0_interval = self.z0_interval[valid_mask]
                self.datum_range = self.datum_range[valid_mask]
            
            if len(self.z0_interval) == 0:
                raise ValueError("No valid data points remaining after cleaning.")
            
            # IMPROVED LOGIC: Use direction-specific results instead of global min/max
            # Find Z0L and Z0U from respective search directions
            self.z0 = self._z0_main
            if len(lower_search_data['z0']) > 0:
                lower_z0_array = np.array(lower_search_data['z0'])
                lower_datum_array = np.array(lower_search_data['datum'])
                z0l_idx = np.argmin(lower_z0_array)
                self.z0l = float(lower_z0_array[z0l_idx])
                self.zl = float(lower_datum_array[z0l_idx])
            else:
                self.z0l = float(self.z0)
                self.zl = float(self.z0)
            
            if len(upper_search_data['z0']) > 0:
                upper_z0_array = np.array(upper_search_data['z0'])
                upper_datum_array = np.array(upper_search_data['datum'])
                z0u_idx = np.argmax(upper_z0_array)
                self.z0u = float(upper_z0_array[z0u_idx])
                self.zu = float(upper_datum_array[z0u_idx])
            else:
                self.z0u = float(self.z0)
                self.zu = float(self.z0)
            
            # Ensure logical ordering: ZL ≤ Z0 ≤ ZU
            if self.zl > self.zu:
                if self.verbose:
                    print(f"Swapping ZL and ZU: ZL was {self.zl:.6f}, ZU was {self.zu:.6f}")
                self.zl, self.zu = self.zu, self.zl
                self.z0l, self.z0u = self.z0u, self.z0l
            
            # Additional validation
            if self.z0l > self.z0u:
                if self.verbose:
                    print(f"Warning: Z0L ({self.z0l:.6f}) > Z0U ({self.z0u:.6f}). Using fallback logic.")
                # Fallback: use global min/max from combined data
                z0l_idx_global = np.argmin(self.z0_interval)
                z0u_idx_global = np.argmax(self.z0_interval)
                self.z0l = float(self.z0_interval[z0l_idx_global])
                self.z0u = float(self.z0_interval[z0u_idx_global])
                self.zl = float(self.datum_range[z0l_idx_global])
                self.zu = float(self.datum_range[z0u_idx_global])
            
            # Store parameters if catching is enabled
            if self.catch:
                self.params = getattr(self.init_egdf, 'params', {}).copy()
                self.params.update({
                    'ZL': self.zl,
                    'Z0L': self.z0l,
                    'Z0': float(self.z0),
                    'Z0U': self.z0u,
                    'ZU': self.zu,
                    'lower_search_points': len(lower_search_data['z0']),
                    'upper_search_points': len(upper_search_data['z0'])
                })
            
            # Verbose output
            if self.verbose:
                print(f"\nInterval Analysis Results:")
                print(f"ZL:  {self.zl:.6f} (datum producing minimum Z0)")
                print(f"Z0L: {self.z0l:.6f} (minimum Z0 value)")
                print(f"Z0:  {self.z0:.6f} (original central point)")
                print(f"Z0U: {self.z0u:.6f} (maximum Z0 value)")
                print(f"ZU:  {self.zu:.6f} (datum producing maximum Z0)")
                print(f"Lower search points: {len(lower_search_data['z0'])}")
                print(f"Upper search points: {len(upper_search_data['z0'])}")
                print("Interval values computed successfully.")
        
        except Exception as e:
            error_msg = f"Error in interval computation: {e}"
            if self.verbose:
                print(error_msg)
            raise RuntimeError(error_msg) from e

    
    def _plot_egdf_intv(self, plot_style='scatter', show_data_points=True, show_grid=True, figsize=(12, 8)):
        """
        Plot interval analysis results showing data range vs Z0 intervals.
        
        Parameters:
        -----------
        plot_style : str, default='smooth'
            Style of the main curve. Options: 'smooth', 'scatter', 'line'
        show_data_points : bool, default=True
            Whether to show individual data points as scatter
        show_grid : bool, default=True
            Whether to show grid lines
        figsize : tuple, default=(12, 8)
            Figure size as (width, height)
        
        Creates a comprehensive visualization with:
        - X-axis: Data range (datum values) with proper limits
        - Y-axis: Z0 interval values
        - Logical legend order: ZL - Z0L - Z0 - Z0U - ZU
        - Robust error handling and fallback options
        """
        try:
            # this plot only works when intervals are computed with _compute_intv_linear_search only
            # check if self.z0_interval and self.datum_range exist else stop here

            import matplotlib.pyplot as plt
            
            # Check if interval computation has been performed
            if not hasattr(self, 'z0_interval') or not hasattr(self, 'datum_range'):
                warnings.warn(
                    "Interval plot data unavailable. This plot requires 'linear_search=True' "
                    "during interval computation. Please re-run with linear_search=True."
                )
                return None, None
            
            if len(self.z0_interval) == 0 or len(self.datum_range) == 0:
                raise ValueError("No data available for plotting. Interval computation may have failed.")
            
            # Validate required attributes
            required_attrs = ['zl', 'z0l', 'z0', 'z0u', 'zu']
            missing_attrs = [attr for attr in required_attrs if not hasattr(self, attr)]
            if missing_attrs:
                raise ValueError(f"Missing required attributes for plotting: {missing_attrs}")
            
            # Validate plot_style
            valid_styles = ['smooth', 'scatter', 'line']
            if plot_style not in valid_styles:
                warnings.warn(f"Invalid plot_style '{plot_style}'. Using 'smooth' instead.")
                plot_style = 'smooth'
            
            # Create the plot with error handling
            try:
                fig, ax = plt.subplots(figsize=figsize)
            except Exception as e:
                print(f"Error creating plot figure: {e}")
                return None, None
            
            # Sort data for smooth line plotting
            try:
                sorted_indices = np.argsort(self.datum_range)
                sorted_datum = self.datum_range[sorted_indices]
                sorted_z0_interval = self.z0_interval[sorted_indices]
            except Exception as e:
                warnings.warn(f"Error sorting data for plotting: {e}")
                sorted_datum = self.datum_range
                sorted_z0_interval = self.z0_interval
            
            # Plot the main curve based on style
            curve_plotted = False
            
            if plot_style == 'smooth' and len(sorted_datum) > 3:
                # Try spline interpolation first
                try:
                    from scipy.interpolate import make_interp_spline
                    spline = make_interp_spline(sorted_datum, sorted_z0_interval, k=min(3, len(sorted_datum)-1))
                    datum_smooth = np.linspace(sorted_datum.min(), sorted_datum.max(), 500)
                    z0_smooth = spline(datum_smooth)
                    
                    # Check for valid interpolation
                    if np.all(np.isfinite(z0_smooth)):
                        ax.plot(datum_smooth, z0_smooth, color='blue', linewidth=2.5, 
                               alpha=0.9, label='Z0 Interval Curve', zorder=3)
                        curve_plotted = True
                    else:
                        raise ValueError("Spline interpolation produced invalid values")
                        
                except Exception as e:
                    if self.verbose:
                        print(f"Spline interpolation failed: {e}, trying linear interpolation")
                    
                    # Fallback to linear interpolation
                    try:
                        from scipy.interpolate import interp1d
                        f_linear = interp1d(sorted_datum, sorted_z0_interval, kind='linear', 
                                          bounds_error=False, fill_value='extrapolate')
                        datum_smooth = np.linspace(sorted_datum.min(), sorted_datum.max(), 300)
                        z0_smooth = f_linear(datum_smooth)
                        
                        # Check for valid interpolation
                        if np.all(np.isfinite(z0_smooth)):
                            ax.plot(datum_smooth, z0_smooth, color='blue', linewidth=2.5, 
                                   alpha=0.9, label='Z0 Interval Curve', zorder=3)
                            curve_plotted = True
                        else:
                            raise ValueError("Linear interpolation produced invalid values")
                            
                    except Exception as e:
                        if self.verbose:
                            print(f"Linear interpolation failed: {e}, using line plot")
            
            # Fallback to line or scatter plot
            if not curve_plotted:
                if plot_style == 'scatter':
                    ax.scatter(sorted_datum, sorted_z0_interval, color='blue', s=60, 
                              alpha=0.8, label='Z0 Interval Points', zorder=3)
                else:  # line plot
                    ax.plot(sorted_datum, sorted_z0_interval, 'b-', linewidth=2, 
                           alpha=0.8, label='Z0 Interval', zorder=3, marker='o', markersize=4)
            
            # Add individual data points if requested
            if show_data_points and plot_style != 'scatter':
                try:
                    ax.scatter(sorted_datum, sorted_z0_interval, marker='x', s=40, 
                              color='gray', alpha=0.8, linewidth=1.5, zorder=4, 
                              label='Data Points')
                except Exception as e:
                    warnings.warn(f"Error plotting data points: {e}")
            
            # Store legend handles and labels in logical order
            legend_handles = []
            legend_labels = []
            
            # Add critical lines and points with error handling - in logical order
            try:
                # 1. ZL (leftmost vertical line)
                zl_line = ax.axvline(x=self.zl, color='red', linestyle='--', linewidth=2, 
                                   alpha=0.8, zorder=2)
                legend_handles.append(zl_line)
                legend_labels.append(f'ZL = {self.zl:.4f}')
                
                # 2. Z0L (horizontal line for minimum Z0)
                z0l_line = ax.axhline(y=self.z0l, color='purple', linestyle=':', linewidth=2, 
                                    alpha=0.7, zorder=1)
                legend_handles.append(z0l_line)
                legend_labels.append(f'Z0L = {self.z0l:.4f}')
                
                # 3. Z0 (central vertical line)
                z0_line = ax.axvline(x=self.z0, color='green', linestyle='-', linewidth=2, 
                                   alpha=0.8, zorder=2)
                legend_handles.append(z0_line)
                legend_labels.append(f'Z0 = {self.z0:.4f}')
                
                # 4. Z0U (horizontal line for maximum Z0)
                z0u_line = ax.axhline(y=self.z0u, color='brown', linestyle=':', linewidth=2, 
                                    alpha=0.7, zorder=1)
                legend_handles.append(z0u_line)
                legend_labels.append(f'Z0U = {self.z0u:.4f}')
                
                # 5. ZU (rightmost vertical line)
                zu_line = ax.axvline(x=self.zu, color='orange', linestyle='--', linewidth=2, 
                                   alpha=0.8, zorder=2)
                legend_handles.append(zu_line)
                legend_labels.append(f'ZU = {self.zu:.4f}')
                
                # Add critical points in logical order
                zl_point = ax.scatter([self.zl], [self.z0l], marker='o', s=150, color='red', 
                                    edgecolor='black', linewidth=2, zorder=5)
                legend_handles.append(zl_point)
                legend_labels.append('(ZL, Z0L)')
                
                z0_point = ax.scatter([self.z0], [self.z0], marker='s', s=150, color='green', 
                                    edgecolor='black', linewidth=2, zorder=5)
                legend_handles.append(z0_point)
                legend_labels.append('(Z0, Z0)')
                
                zu_point = ax.scatter([self.zu], [self.z0u], marker='o', s=150, color='orange', 
                                    edgecolor='black', linewidth=2, zorder=5)
                legend_handles.append(zu_point)
                legend_labels.append('(ZU, Z0U)')
                          
            except Exception as e:
                warnings.warn(f"Error plotting critical lines and points: {e}")
            
            # Set axis limits with robust padding
            try:
                # Use ZL and ZU for x-axis limits with padding
                if hasattr(self, 'zl') and hasattr(self, 'zu'):
                    x_range = self.zu - self.zl
                    if x_range == 0:  # Ensure minimum padding
                        x_range = 0.1  # Minimum range for padding
                    x_padding = x_range * 0.2  # 20% padding or minimum 0.1
                    ax.set_xlim(self.zl - x_padding, self.zu + x_padding)
                
                # Use Z0L and Z0U for y-axis limits with padding
                if hasattr(self, 'z0l') and hasattr(self, 'z0u'):
                    y_range = self.z0u - self.z0l
                    if y_range == 0:  # Ensure minimum padding
                        y_range = 0.1
                    y_padding = y_range * 0.2  # 20% padding or minimum
                    ax.set_ylim(self.z0l - y_padding, self.z0u + y_padding)
                    
            except Exception as e:
                warnings.warn(f"Error setting axis limits: {e}")
            
            # Set labels and title
            try:
                ax.set_xlabel('Datum Value', fontsize=10, fontweight='bold')
                ax.set_ylabel('Z0 Value', fontsize=10, fontweight='bold')
                ax.set_title('EGDF Interval Analysis\n' + 
                            f'Points analyzed: {len(self.z0_interval)} | Style: {plot_style}', 
                            fontsize=10)
                
                # Add grid if requested
                if show_grid:
                    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                
                # Add legend with logical order
                try:
                    # Combine all legend handles and labels
                    ax.legend(legend_handles, legend_labels, bbox_to_anchor=(1.05, 1), 
                             loc='upper left', fontsize=10, title='Interval Points (ZL → ZU)')
                except Exception as e:
                    try:
                        ax.legend(legend_handles, legend_labels, fontsize=10)  # Fallback to default legend
                    except Exception as e2:
                        warnings.warn(f"Could not create legend: {e2}")
                    
            except Exception as e:
                warnings.warn(f"Error setting plot labels and formatting: {e}")
            
            # Adjust layout
            try:
                plt.tight_layout()
            except Exception as e:
                warnings.warn(f"Error adjusting layout: {e}")
            
            # Show plot
            try:
                plt.show()
            except Exception as e:
                warnings.warn(f"Error displaying plot: {e}")
            
            return fig, ax
            
        except Exception as e:
            error_msg = f"Error in plotting interval analysis: {e}"
            if self.verbose:
                print(error_msg)
            warnings.warn(error_msg)
            return None, None

    def _get_intv(self, decimals: int=2) -> dict:
        """
        Get the interval values ZL, Z0L, Z0, Z0U, and ZU after fitting.
        
        Returns:
        --------
        dict
            Dictionary containing the interval values.
        """
        if not hasattr(self, 'z0l') or not hasattr(self, 'z0u'):
            raise ValueError("Interval values have not been computed yet. First fit the model for interval analysis.")

        return {
            'LB': float(np.round(self.LB, decimals)),
            'LSB': float(np.round(self.LSB, decimals)) if hasattr(self, 'LSB') else None,
            'DLB': float(np.round(self.DLB, decimals)),
            'CLB': float(np.round(self.CLB, decimals)) if hasattr(self, 'CLB') else None,
            'ZL': float(np.round(self.zl, decimals)),
            'Z0L': float(np.round(self.z0l, decimals)),
            'Z0': float(np.round(self.z0, decimals)),
            'Z0U': float(np.round(self.z0u, decimals)),
            'ZU': float(np.round(self.zu, decimals)),
            'CUB': float(np.round(self.CUB, decimals)) if hasattr(self, 'CUB') else None,
            'DUB': float(np.round(self.DUB, decimals)),
            'USB': float(np.round(self.USB, decimals)) if hasattr(self, 'USB') else None,
            'UB': float(np.round(self.UB, decimals)),
        }
    
    def _plot_egdf(self, plot_type: str = 'marginal', plot_smooth: bool = True, bounds: bool = True, 
                   derivatives: bool = False, intervals: bool = True, show_all_bounds: bool = False, 
                   figsize: tuple = (12, 8)):
        """
        Enhanced plotting for marginal analysis with LSB, USB, clustering visualization, and interval analysis.
        
        Parameters:
        -----------
        plot_type : str, default='marginal'
            Type of plot: 'marginal', 'egdf', 'pdf', 'both', 'clusters'
        plot_smooth : bool, default=True
            Whether to plot smooth curves
        bounds : bool, default=True
            Whether to show bounds (LB, UB, DLB, DUB, LSB, USB, CLB, CUB)
        derivatives : bool, default=False
            Whether to show derivative analysis plot
        intervals : bool, default=True
            Whether to show interval analysis points (ZL, Z0L, Z0, Z0U, ZU)
        show_all_bounds : bool, default=False
            Whether to show all available bounds (CLB, CUB, LSB, USB, etc.)
        figsize : tuple, default=(12, 8)
            Figure size
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        if not self.catch:
            print("Plot is not available with argument catch=False")
            return
        
        if not hasattr(self.init_egdf, '_fitted') or not self.init_egdf._fitted:
            raise RuntimeError("Must fit marginal analysis before plotting.")
        
        # Validate plot_type
        valid_types = ['marginal', 'egdf', 'pdf', 'both']
        if plot_type not in valid_types:
            raise ValueError(f"plot_type must be one of {valid_types}")
        
        if derivatives:
            self._plot_derivatives(figsize=figsize)
            return
        
        # Adjust figure size to accommodate external summary
        if intervals:
            figsize = (figsize[0] + 3, figsize[1])  # Add space for summary
        
        # Create single plot with dual y-axes
        fig, ax1 = plt.subplots(figsize=figsize)
        
        # Get data
        x_points = self.init_egdf.data
        egdf_vals = self.init_egdf.params.get('egdf')
        pdf_vals = self.init_egdf.params.get('pdf')
        wedf_vals = self.init_egdf.params.get('wedf')
        
        # Plot EGDF on primary y-axis
        if plot_type in ['marginal', 'egdf', 'both']:
            if plot_smooth and hasattr(self.init_egdf, 'egdf_points') and self.init_egdf.egdf_points is not None:
                ax1.plot(x_points, egdf_vals, 'o', color='blue', label='EGDF', markersize=4)
                ax1.plot(self.init_egdf.di_points_n, self.init_egdf.egdf_points, 
                        color='blue', linestyle='-', linewidth=2, alpha=0.8)
            else:
                ax1.plot(x_points, egdf_vals, 'o-', color='blue', label='EGDF', 
                        markersize=4, linewidth=2, alpha=0.8)
        
        # Plot WEDF if available
        if wedf_vals is not None and plot_type in ['marginal', 'egdf', 'both']:
            ax1.plot(x_points, wedf_vals, 's', color='lightblue', 
                    label='WEDF', markersize=3, alpha=0.8)
        
        ax1.set_xlabel('Data Points', fontsize=12, fontweight='bold')
        ax1.set_ylabel('EGDF', color='blue', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_ylim(0, 1)
        
        # Create secondary y-axis for PDF
        ax2 = ax1.twinx()
        
        if plot_type in ['marginal', 'pdf', 'both']:
            if plot_smooth and hasattr(self.init_egdf, 'pdf_points') and self.init_egdf.pdf_points is not None:
                ax2.plot(x_points, pdf_vals, 'o', color='red', label='PDF', markersize=4)
                ax2.plot(self.init_egdf.di_points_n, self.init_egdf.pdf_points, 
                        color='red', linestyle='-', linewidth=2, alpha=0.8)
                max_pdf = np.max(self.init_egdf.pdf_points)
            else:
                ax2.plot(x_points, pdf_vals, 'o-', color='red', label='PDF', 
                        markersize=4, linewidth=2, alpha=0.8)
                max_pdf = np.max(pdf_vals) if pdf_vals is not None else 1
        
        ax2.set_ylabel('PDF', color='red', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='red')
        if 'max_pdf' in locals():
            ax2.set_ylim(0, max_pdf * 1.1)
        
        # Add bounds only if bounds=True
        if bounds:
            self._add_bounds(ax1, show_all_bounds=show_all_bounds)
        
        # Add marginal points (Z0 always, others only if bounds=True)
        self._add_marginal_points(ax1, bounds=bounds)
        
        # Add interval analysis points if intervals=True and they exist
        if intervals:
            summary_text = self._add_interval_points_external(ax1, ax2, fig)
        
        # Set xlim to DLB-DUB range
        if hasattr(self.init_egdf, 'DLB') and hasattr(self.init_egdf, 'DUB'):
            # 5% data pad on either side
            pad = (self.init_egdf.DUB - self.init_egdf.DLB) * 0.05
            ax1.set_xlim(self.init_egdf.DLB - pad, self.init_egdf.DUB + pad)
            ax2.set_xlim(self.init_egdf.DLB - pad, self.init_egdf.DUB + pad)
    
        # Add shaded regions for bounds only if bounds=True
        if bounds and hasattr(self.init_egdf, 'LB') and hasattr(self.init_egdf, 'UB'):
            if self.init_egdf.LB is not None:
                ax1.axvspan(self.init_egdf.DLB, self.init_egdf.LB, alpha=0.15, color='purple')
            if self.init_egdf.UB is not None:
                ax1.axvspan(self.init_egdf.UB, self.init_egdf.DUB, alpha=0.15, color='brown')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        
        # Position legend
        total_items = len(labels1) + len(labels2)
        if total_items > 8:
            ax1.legend(lines1 + lines2, labels1 + labels2, 
                      bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        else:
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
        
        # Set title
        plt.title('Interval Analysis', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _add_bounds(self, ax, show_all_bounds=False):
        """
        Add bounds to the plot with option to show all available bounds.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            Axis to add bounds to
        show_all_bounds : bool, default=False
            Whether to show all available bounds
        """
        # Standard bounds (always shown if bounds=True)
        bounds_config = [
            ('DLB', 'darkgreen', '-'),
            ('DUB', 'darkgreen', '-'),
            ('LB', 'red', '--'),
            ('UB', 'red', '--')
        ]
        
        # Additional bounds (shown only if show_all_bounds=True)
        if show_all_bounds:
            additional_bounds = [
                ('LSB', 'purple', ':'),
                ('USB', 'purple', ':'),
                ('CLB', 'orange', '-.'),
                ('CUB', 'orange', '-.')
            ]
            bounds_config.extend(additional_bounds)
        
        for bound_name, color, linestyle in bounds_config:
            if hasattr(self.init_egdf, bound_name):
                bound_value = getattr(self.init_egdf, bound_name)
                if bound_value is not None:
                    ax.axvline(x=bound_value, color=color, linestyle=linestyle, 
                              linewidth=1, alpha=0.7, label=f'{bound_name}={bound_value:.3f}')
    
    def _add_interval_points_external(self, ax1, ax2, fig):
        """
        Add interval analysis points with external summary on the right side.
        
        Parameters:
        -----------
        ax1 : matplotlib.axes.Axes
            Primary axis (for EGDF)
        ax2 : matplotlib.axes.Axes
            Secondary axis (for PDF)
        fig : matplotlib.figure.Figure
            Figure object
        """
        # Check if interval analysis has been performed
        interval_attrs = ['zl', 'z0l', 'z0', 'z0u', 'zu']
        available_intervals = [attr for attr in interval_attrs if hasattr(self, attr)]
        
        if not available_intervals:
            if self.verbose:
                print("No interval analysis points available. Run interval analysis first.")
            return ""
        
        # Define interval point configurations with linewidth=1
        interval_configs = [
            ('zl', 'ZL', 'red', '--', 'o', 10),        # ZL: red dashed line, circle marker
            ('z0l', 'Z0L', 'purple', ':', 's', 8),     # Z0L: purple dotted line, square marker  
            ('z0', 'Z0', 'green', '-', '^', 12),       # Z0: green solid line, triangle marker
            ('z0u', 'Z0U', 'brown', ':', 's', 8),      # Z0U: brown dotted line, square marker
            ('zu', 'ZU', 'orange', '--', 'o', 10)      # ZU: orange dashed line, circle marker
        ]
        
        # Get axis limits for positioning dots
        y1_min, y1_max = ax1.get_ylim()
        y2_min, y2_max = ax2.get_ylim()
        
        # Position for dots on x-axis (slightly below the plot area)
        dot_y_position_ax1 = y1_min - (y1_max - y1_min) * 0.08
        dot_y_position_ax2 = y2_min - (y2_max - y2_min) * 0.08
        
        # Store values for summary
        interval_summary = []
        
        for attr_name, label, color, linestyle, marker, size in interval_configs:
            if hasattr(self, attr_name):
                value = getattr(self, attr_name)
                
                # Add vertical line with linewidth=1
                line = ax1.axvline(x=value, color=color, linestyle=linestyle, 
                                  linewidth=1, alpha=0.8, zorder=3)
                
                # Add highlight dot on x-axis (primary axis)
                dot1 = ax1.scatter([value], [dot_y_position_ax1], 
                                  marker=marker, s=size**2, color=color, 
                                  edgecolor='black', linewidth=1.5, 
                                  zorder=5, clip_on=False)
                
                # Add corresponding dot on secondary axis
                dot2 = ax2.scatter([value], [dot_y_position_ax2], 
                                  marker=marker, s=size**2, color=color, 
                                  edgecolor='black', linewidth=1.5, 
                                  zorder=5, clip_on=False, alpha=0.7)
                
                # Store for summary
                interval_summary.append((label, value, color))
        
        # Add filled spans if all points are available
        if len(available_intervals) == 5:
            # Add filled span for typical data interval (ZL to ZU) - blue
            ax1.axvspan(self.zl, self.zu, alpha=0.15, color='blue', 
                       label='Interval of Typical Data', zorder=1)
            
            # Add filled span for tolerance interval (Z0L to Z0U) - light green
            # Only add if there's actually a measurable tolerance interval
            tolerance_interval = self.z0u - self.z0l
            if tolerance_interval > 1e-10:  # Only show if tolerance interval is meaningful
                ax1.axvspan(self.z0l, self.z0u, alpha=0.2, color='lightgreen', 
                           label='Tolerance Interval', zorder=2)
            
            # Mark Z0L and Z0U values on the EGDF curve if possible
            if hasattr(self, 'z0_interval') and hasattr(self, 'datum_range'):
                try:
                    # Interpolate EGDF values at ZL and ZU
                    if hasattr(self.init_egdf, 'egdf_points'):
                        zl_egdf = np.interp(self.zl, self.init_egdf.di_points_n, self.init_egdf.egdf_points)
                        zu_egdf = np.interp(self.zu, self.init_egdf.di_points_n, self.init_egdf.egdf_points)
                    else:
                        zl_egdf = np.interp(self.zl, self.init_egdf.data, self.init_egdf.params.get('egdf', []))
                        zu_egdf = np.interp(self.zu, self.init_egdf.data, self.init_egdf.params.get('egdf', []))
                    
                    # Mark these points on the EGDF curve
                    ax1.scatter([self.zl, self.zu], [zl_egdf, zu_egdf], 
                               marker='*', s=150, color=['red', 'orange'], 
                               edgecolor='black', linewidth=2, zorder=6,
                               label='Interval Points on EGDF')
                    
                except Exception as e:
                    if self.verbose:
                        print(f"Could not mark interval points on EGDF curve: {e}")
        
        # Create external summary text
        if len(available_intervals) == 5:
            tolerance_interval = self.z0u - self.z0l
            typical_data_interval = self.zu - self.zl
            
            summary_lines = [
                "Interval Analysis Summary:",
                "─" * 25,
                f"ZL  = {self.zl:.4f}",
                f"Z0L = {self.z0l:.4f}",
                f"Z0  = {self.z0:.4f}",
                f"Z0U = {self.z0u:.4f}",
                f"ZU  = {self.zu:.4f}",
                "",
                "Interval Ranges:",
                "─" * 16,
                f"Tolerance Interval: {tolerance_interval:.6f}",
                f"Interval of Typical Data: {typical_data_interval:.4f}",
                "",
                "Definitions:",
                "─" * 12,
                "• Tolerance Interval: Z0U - Z0L",
                "• Interval of Typical Data: ZU - ZL",
            ]
        else:
            summary_lines = [
                "Interval Analysis (Partial):",
                "─" * 28
            ]
            for label, value, color in interval_summary:
                summary_lines.append(f"{label} = {value:.4f}")
        
        summary_text = "\n".join(summary_lines)
        
        # Add external text box to the right of the plot
        fig.text(0.85, 0.02, summary_text, 
                 fontsize=10, verticalalignment='bottom', horizontalalignment='left',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', 
                          alpha=0.9, edgecolor='gray', linewidth=1),
                 transform=fig.transFigure)
        
        # Extend axis limits slightly to accommodate the dots
        ax1.set_ylim(dot_y_position_ax1 - (y1_max - y1_min) * 0.05, y1_max * 1.05)
        ax2.set_ylim(dot_y_position_ax2 - (y2_max - y2_min) * 0.05, y2_max * 1.05)
        
        if self.verbose and available_intervals:
            print(f"Added {len(available_intervals)} interval analysis points to plot: {available_intervals}")
        
        return summary_text
    
    def _is_homogeneous(self):
        """
        Check if the data is homogeneous.
        Returns True if homogeneous, False otherwise.
        """
        ih = DataHomogeneity(self.init_egdf, catch=self.catch, verbose=self.verbose)
        is_homogeneous = ih.test_homogeneity()

        if self.catch:
            self.init_egdf.params['is_homogeneous'] = is_homogeneous
        return is_homogeneous
    
    def _fit_egdf_intv(self, plot=True):
        try:
            if self.verbose:
                print("\n\nFitting EGDF Interval Analysis...")
                
            # get initial EGDF
            self._get_initial_egdf()
            self.params = getattr(self.init_egdf, 'params', {}).copy()

            # homogeneous check
            self.h = self._is_homogeneous()

            if self.h:
                if self.verbose:
                    print("Data is homogeneous. Using homogeneous data for interval analysis.")
            else:
                if self.verbose:
                    print("Data is heterogeneous. Need to estimate cluster bounds to find main cluster.")
            
            # h check
            if self.h == False and self.estimate_cluster_bounds == False and self.get_clusters == True:
                warnings.warn("Data is heterogeneous but estimate_cluster_bounds is False. "
                            "Consider setting 'estimate_cluster_bounds=True' and 'get_clusters=True' to find main cluster bounds and main cluster.")

            # optional data sampling bounds
            if self.estimate_sample_bounds:
                self._get_data_sample_bounds()

            # cluster bounds
            if self.estimate_cluster_bounds:
                self._get_data_sample_clusters() # if get_clusters is True, it will estimate cluster bounds

            # get Z0 of the base sample
            self._z0_main = self._get_z0(self.init_egdf)

            if self.verbose:
                print("Initiating EGDF Interval Analysis...")

            # compute interval values
            if self.linear_search:
                if self.verbose:
                    print("Using linear search for interval analysis...")
                self._compute_intv_linear_search()
            else:
                if self.verbose:
                    print("Using optimized search for interval analysis...")
                self._compute_intv()

            # plot if requested
            if plot:
                self._plot_egdf(plot_type='both', plot_smooth=True, bounds=True,
                                derivatives=False, intervals=True,
                                show_all_bounds=True, figsize=(12, 8))

            if self.verbose:
                print("EGDF Interval Analysis fitted successfully.")
        
        except Exception as e:
            if self.verbose:
                print(f"Error occurred during fitting: {e}")
