"""
Interval Estimation Engine (IntveEngine) - Fresh Implementation with Improved Logic
Core Logic: Extend data with single datum and track Z0 variations with ordering constraint
"""

import numpy as np
import warnings
from typing import Union, Dict, Tuple, Optional, List
from machinegnostics.magcal import EGDF, ELDF
import logging
from machinegnostics.magcal.util.logging import get_logger

class IntveEngine:
    """
    Z0-Based Interval Estimation Engine with Ordering Constraint Validation
    
    This class implements a novel approach to interval estimation using Z0 (gnostic mode) variations.
    It extends the original data with single datum points across the [LB, UB] range and tracks how
    the Z0 value changes to identify optimal interval boundaries that satisfy the ordering constraint:
    ZL < Z0L < Z0 < Z0U < ZU
    
    **Core Methodology:**
    
    1. **Data Extension Strategy**: For each potential datum value in [LB, UB], extend the original 
       data with that single datum and recompute the Z0 (gnostic mode).
       
    2. **Z0 Variation Tracking**: Monitor how Z0 changes as different datum values are added:
       - Z0 decreases when certain datum values are added → minimum gives Z0L, ZL
       - Z0 increases when other datum values are added → maximum gives Z0U, ZU
       
    3. **Interval Identification**:
       - **ZL**: Datum value that produces minimum Z0 (Z0L)
       - **Z0L**: Minimum Z0 value achieved
       - **ZU**: Datum value that produces maximum Z0 (Z0U) 
       - **Z0U**: Maximum Z0 value achieved
       - **Tolerance Interval**: [Z0L, Z0U] - range of Z0 variations
       - **Typical Data Interval**: [ZL, ZU] - range of datum values causing extrema
    
    4. **Ordering Constraint**: Ensures ZL < Z0L < Z0_original < Z0U < ZU for valid intervals
    
    **Key Features:**
    
    - **Adaptive Search**: Dense sampling near Z0, sparse sampling toward boundaries
    - **Convergence Detection**: Early stopping when Z0 variations stabilize
    - **Robust Fallback**: Handles cases where strict ordering constraint cannot be satisfied
    - **Dual DF Support**: Works with both EGDF and ELDF objects
    - **Comprehensive Validation**: Multiple validation checks and constraint enforcement
    - **Rich Diagnostics**: Detailed search statistics and quality metrics
    
    **Applications:**
    
    - Quality control interval estimation
    - Uncertainty quantification in manufacturing
    - Process capability analysis
    - Statistical tolerance design
    - Risk assessment and decision making
    
    Parameters
    ----------
    df_object : Union[EGDF, ELDF]
        Fitted distribution function object with available Z0 (gnostic mode).
        Must be already fitted using df_object.fit().
        
    n_points_per_direction : int, optional (default=1000)
        Number of search points to generate in each direction from Z0.
        Higher values provide more precise interval estimation but increase computation time.
        - Recommended: 500-2000 for most applications
        - Minimum: 50 (automatically enforced)
        
    dense_zone_fraction : float, optional (default=0.4)
        Fraction of the search range to sample densely near Z0.
        Controls the balance between local precision and boundary exploration.
        - Range: [0.1, 0.8] (automatically clipped)
        - 0.4 means 40% of range near Z0 gets dense sampling
        
    dense_points_fraction : float, optional (default=0.7)
        Fraction of total points to place in the dense zone.
        Higher values focus more search effort near Z0.
        - Range: [0.5, 0.9] (automatically clipped)
        - 0.7 means 70% of points in dense zone, 30% toward boundaries
        
    convergence_window : int, optional (default=15)
        Number of recent points to check for Z0 convergence.
        Enables early stopping when Z0 variations stabilize.
        - Minimum: 5 (automatically enforced)
        - Larger windows are more conservative
        
    convergence_threshold : float, optional (default=1e-7)
        Standard deviation threshold for detecting Z0 convergence.
        Lower values require more stable convergence before stopping.
        - Typical range: 1e-9 to 1e-5
        
    min_search_points : int, optional (default=30)
        Minimum number of points to search before checking convergence.
        Prevents premature stopping in early search phases.
        - Minimum: 10 (automatically enforced)
        
    boundary_margin_factor : float, optional (default=0.001)
        Safety margin from LB/UB boundaries as fraction of range.
        Prevents numerical issues near boundaries.
        - Minimum: 1e-6 (automatically enforced)
        - 0.001 means 0.1% margin from each boundary
        
    extrema_search_tolerance : float, optional (default=1e-6)
        Tolerance for identifying valid extrema that satisfy ordering constraint.
        Used in numerical comparisons during extrema validation.
        
    verbose : bool, optional (default=False)
        Enable detailed progress reporting and diagnostic output.
        Useful for debugging and understanding the search process.
    
    Attributes
    ----------
    zl : float
        Datum value that produces minimum Z0 (left boundary of typical data interval).
        
    z0l : float
        Minimum Z0 value achieved (left boundary of tolerance interval).
        
    zu : float
        Datum value that produces maximum Z0 (right boundary of typical data interval).
        
    z0u : float
        Maximum Z0 value achieved (right boundary of tolerance interval).
        
    z0 : float
        Original Z0 value from the fitted DF object. Accessible as obj.z0 for convenience.
        
    tolerance_interval : float
        Width of tolerance interval (Z0U - Z0L).
        
    typical_data_interval : float
        Width of typical data interval (ZU - ZL).
        
    params : dict
        Comprehensive parameter dictionary containing:
        - Configuration settings
        - Search results and statistics
        - Quality metrics and validation results
        - Timing information
        
    search_results : dict
        Detailed search tracking with 'lower' and 'upper' direction results:
        - datum_values: List of tested datum values
        - z0_values: Corresponding Z0 values
        - success_flags: Success/failure status for each attempt
    
    Examples
    --------
    Basic usage with ELDF:
    
    >>> from machinegnostics.magcal import ELDF
    >>> from machinegnostics.magcal.gdf.intv_engine import IntveEngine
    >>> 
    >>> # Create and fit ELDF
    >>> data = np.array([18, 19, 20, 21, 22])
    >>> eldf = ELDF(data, LB=15, UB=25)
    >>> eldf.fit()
    >>> 
    >>> # Create and fit interval engine
    >>> intve = IntveEngine(eldf, verbose=True)
    >>> intve.fit(plot=True)
    >>> 
    >>> # Access results
    >>> print(f"Z0: {intve.z0}")
    >>> print(f"Tolerance interval: [{intve.z0l:.4f}, {intve.z0u:.4f}]")
    >>> print(f"Typical data interval: [{intve.zl:.4f}, {intve.zu:.4f}]")
    >>> 
    >>> # Get complete results dictionary
    >>> intervals = intve.get_intervals()
    
    Advanced configuration:
    
    >>> # High-precision search with custom parameters
    >>> intve = IntveEngine(
    ...     eldf, 
    ...     n_points_per_direction=2000,
    ...     dense_zone_fraction=0.3,
    ...     convergence_threshold=1e-8,
    ...     verbose=True
    ... )
    >>> intve.fit()
    
    Working with search results:
    
    >>> # Access detailed search data
    >>> lower_data = intve.search_results['lower']
    >>> upper_data = intve.search_results['upper']
    >>> 
    >>> # Check ordering constraint satisfaction
    >>> ordering_valid = (intve.zl < intve.z0l < intve.z0 < intve.z0u < intve.zu)
    >>> print(f"Ordering constraint satisfied: {ordering_valid}")
    
    Methods
    -------
    fit(plot=False, update_df_params=True)
        Perform interval estimation with optional plotting and DF parameter updates.
        
    get_intervals(decimals=6)
        Return interval results as formatted dictionary.
        
    plot(figsize=(12, 8), plot_distribution=False, eldf_plot=True)
        Create visualization of interval estimation results.
        
    Notes
    -----
    **Theoretical Foundation:**
    
    The method is based on the principle that adding specific datum values to a dataset
    will cause predictable changes in the Z0 (gnostic mode). By systematically exploring
    these changes, we can identify critical boundaries that define meaningful intervals
    for quality control and process analysis.
    
    **Ordering Constraint Interpretation:**
    
    The constraint ZL < Z0L < Z0 < Z0U < ZU ensures that:
    - ZL and ZU represent extreme datum values that still produce meaningful Z0 changes
    - Z0L and Z0U represent the range of Z0 sensitivity
    - The original Z0 lies between these extremes, indicating stability
    
    **Performance Considerations:**
    
    - Computation time scales with n_points_per_direction and data size
    - Dense sampling near Z0 is most critical for accuracy
    - Convergence detection can significantly reduce computation time
    - Memory usage is generally modest (< 100MB for typical problems)
    
    **Numerical Stability:**
    
    - Uses adaptive tolerance relaxation for extended DF fitting
    - Implements fallback methods for difficult cases  
    - Applies boundary margins to prevent numerical issues
    - Validates all intermediate results
    
    **Quality Indicators:**
    
    - ordering_constraint_satisfied: Primary validity indicator
    - search_statistics.success_rate: Measure of numerical stability
    - interval_quality.z0_stability: Measure of Z0 sensitivity
    - fit_time: Performance indicator
    
    References
    ----------
    Based on the theoretical framework of Machine Gnostics and the principles of
    gnostic mode analysis for industrial quality control applications.
    
    See Also
    --------
    ELDF : Empirical Log Density Function for univariate data
    EGDF : Empirical Goodness Distribution Function
    Z0Estimator : Z0 estimation utilities
    """
    
    def __init__(self, 
                 df_object: Union[EGDF, ELDF],
                 n_points_per_direction: int = 1000,
                 dense_zone_fraction: float = 0.4,
                 dense_points_fraction: float = 0.7,
                 convergence_window: int = 15,
                 convergence_threshold: float = 1e-7,
                 min_search_points: int = 30,
                 boundary_margin_factor: float = 0.001,
                 extrema_search_tolerance: float = 1e-6,
                 verbose: bool = False):
        """
        Initialize interval estimation engine.
        
        Parameters:
        -----------
        df_object : EGDF or ELDF
            Fitted distribution function object with known Z0
        n_points_per_direction : int
            Number of search points in each direction from Z0
        dense_zone_fraction : float
            Fraction of range to sample densely near Z0
        dense_points_fraction : float  
            Fraction of points to place in dense zone
        convergence_window : int
            Window size for checking Z0 convergence
        convergence_threshold : float
            Threshold for Z0 convergence detection
        min_search_points : int
            Minimum points to search before stopping
        boundary_margin_factor : float
            Safety margin from boundaries as fraction of range
        extrema_search_tolerance : float
            Tolerance for finding valid extrema that satisfy ordering constraint
        verbose : bool
            Enable verbose output
        """
        
        # Set verbose first
        self.verbose = verbose
        
        # Configuration
        self.n_points_per_direction = max(n_points_per_direction, 50)
        self.dense_zone_fraction = np.clip(dense_zone_fraction, 0.1, 0.8)
        self.dense_points_fraction = np.clip(dense_points_fraction, 0.5, 0.9)
        self.convergence_window = max(convergence_window, 5)
        self.convergence_threshold = convergence_threshold
        self.min_search_points = max(min_search_points, 10)
        self.boundary_margin_factor = max(boundary_margin_factor, 1e-6)
        self.extrema_search_tolerance = extrema_search_tolerance

        # logger setup
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if verbose else logging.WARNING)
        self.logger.debug(f"{self.__class__.__name__} initialized::")
        
        # Initialize params dictionary
        self.params = {}
        
        # Results
        self._reset_results()

        # Validate and extract properties from DF object
        self._validate_and_extract_properties(df_object)
        
        # Search tracking
        self.search_results = {
            'lower': {'datum_values': [], 'z0_values': [], 'success_flags': []},
            'upper': {'datum_values': [], 'z0_values': [], 'success_flags': []}
        }
        
        self._fitted = False
        
        # Store initialization parameters
        self._store_initialization_params()
        
        if self.verbose:
            self._print_initialization_info()
    
    def _store_initialization_params(self):
        """Store initialization parameters in params dictionary."""
        self.logger.info("Storing initialization parameters.")  
        self.params.update({
            # Configuration parameters
            'n_points_per_direction': self.n_points_per_direction,
            'dense_zone_fraction': self.dense_zone_fraction,
            'dense_points_fraction': self.dense_points_fraction,
            'convergence_window': self.convergence_window,
            'convergence_threshold': self.convergence_threshold,
            'min_search_points': self.min_search_points,
            'boundary_margin_factor': self.boundary_margin_factor,
            'extrema_search_tolerance': self.extrema_search_tolerance,
            'verbose': self.verbose,
            
            # DF object information
            'df_type': getattr(self, 'df_type', None),
            'original_data_size': len(getattr(self, 'original_data', [])),
            'LB': getattr(self, 'LB', None),
            'UB': getattr(self, 'UB', None),
            'z0_original': getattr(self, 'z0_original', None),
            
            # Status
            'fitted': self._fitted,
            'initialization_time': np.datetime64('now')
        })
    
    def _validate_and_extract_properties(self, df_object):
        """Extract and validate properties from DF object."""
        self.logger.info("Validating and extracting properties from DF object.")
        
        # Check if object is fitted
        if not hasattr(df_object, '_fitted') or not df_object._fitted:
            self.logger.error("Distribution function object must be fitted first")
            raise ValueError("Distribution function object must be fitted first")
        
        # Check if Z0 is available
        if not hasattr(df_object, 'z0') or df_object.z0 is None:
            self.logger.error("Z0 (gnostic mode) not available. Fit Z0 first.")
            raise ValueError("Z0 (gnostic mode) not available. Fit Z0 first.")
        
        # Store reference and extract basic properties
        self.df_object = df_object
        self.original_data = np.array(df_object.data)
        self.LB = float(df_object.LB)
        self.UB = float(df_object.UB)
        self.z0 = float(df_object.z0)
        
        # Validate bounds
        if self.LB >= self.UB:
            self.logger.error(f"Invalid bounds: LB ({self.LB}) >= UB ({self.UB})")
            raise ValueError(f"Invalid bounds: LB ({self.LB}) >= UB ({self.UB})")
        
        # Validate Z0 within bounds
        if not (self.LB <= self.z0 <= self.UB):
            self.logger.warning(f"Z0 ({self.z0:.6f}) outside bounds [{self.LB:.6f}, {self.UB:.6f}]")
        
        # Determine DF type
        if isinstance(df_object, EGDF):
            self.df_type = 'EGDF'
        elif isinstance(df_object, ELDF):
            self.df_type = 'ELDF'
        else:
            class_name = df_object.__class__.__name__
            if 'EGDF' in class_name:
                self.df_type = 'EGDF'
            elif 'ELDF' in class_name:
                self.df_type = 'ELDF'  
            else:
                self.logger.error(f"Unsupported distribution type: {class_name}")
                raise ValueError(f"Unsupported distribution type: {class_name}")
        
        # Extract DF creation parameters
        self._extract_df_parameters()
        
        # Update params with extracted information
        self.params.update({
            'df_type': self.df_type,
            'original_data_size': len(self.original_data),
            'LB': self.LB,
            'UB': self.UB,
            'z0_original': self.z0,
            'data_range': [float(np.min(self.original_data)), float(np.max(self.original_data))],
            'bounds_range': self.UB - self.LB
        })
    
    def _extract_df_parameters(self):
        """Extract parameters needed to create new DF instances."""
        self.logger.info("Extracting DF creation parameters.")
        df = self.df_object
        
        # Safely extract DLB and DUB with validation
        def safe_extract_bound(obj, attr_name, default=None):
            """Safely extract bound with validation."""
            try:
                value = getattr(obj, attr_name, default)
                if value is None:
                    return None
                # Convert to float and validate
                value = float(value)
                if not np.isfinite(value):
                    self.logger.warning(f"{attr_name} is not finite ({value}), using None")
                    return None
                return value
            except (AttributeError, TypeError, ValueError) as e:
                self.logger.warning(f"Could not extract {attr_name}: {e}, using None")
                return None
        
        # Extract bounds safely
        self.DLB = safe_extract_bound(df, 'DLB')
        self.DUB = safe_extract_bound(df, 'DUB')
        
        # Common parameters with safe extraction
        self.weights = getattr(df, 'weights', None)
        self.data_form = getattr(df, 'data_form', 'a')
        self.homogeneous = getattr(df, 'homogeneous', True)
        self.tolerance = getattr(df, 'tolerance', 1e-9)
        self.max_data_size = getattr(df, 'max_data_size', 1000)
        
        # EGDF specific parameters
        if self.df_type == 'EGDF':
            self.S_opt = getattr(df, 'S_opt', 'auto')
            self.wedf = getattr(df, 'wedf', True)
            self.opt_method = getattr(df, 'opt_method', 'L-BFGS-B')
        
        # Store extracted parameters in params
        self.params.update({
            'DLB': self.DLB,
            'DUB': self.DUB,
            'data_form': self.data_form,
            'homogeneous': self.homogeneous,
            'tolerance': self.tolerance,
            'max_data_size': self.max_data_size,
            'has_weights': self.weights is not None,
            'weights_shape': np.array(self.weights).shape if self.weights is not None else None
        })
        
        if self.df_type == 'EGDF':
            self.params.update({
                'S_opt': self.S_opt,
                'wedf': self.wedf,
                'opt_method': self.opt_method
            })
        
        self.logger.info(f"Extracted parameters:")
        self.logger.info(f"  DLB: {self.DLB}")
        self.logger.info(f"  DUB: {self.DUB}")
        self.logger.info(f"  Data form: {self.data_form}")
        self.logger.info(f"  Homogeneous: {self.homogeneous}")

    def _reset_results(self):
        """Reset all results to initial state."""
        self.logger.info("Resetting results to initial state.") 
        self.zl = None      # Datum value where Z0 is minimum
        self.z0l = None     # Minimum Z0 value
        self.zu = None      # Datum value where Z0 is maximum  
        self.z0u = None     # Maximum Z0 value
        self.tolerance_interval = None
        self.typical_data_interval = None
        
        # Reset results in params
        if hasattr(self, 'params'):
            self.params.update({
                'ZL': None,
                'Z0L': None,
                'ZU': None,
                'Z0U': None,
                'tolerance_interval': None,
                'typical_data_interval': None,
                'tolerance_interval_width': None,
                'typical_data_interval_width': None,
                'fitted': False,
                'fit_time': None,
                'search_statistics': None,
                'ordering_validation': None
            })
        
    def _print_initialization_info(self):
        """Print initialization information."""
        self.logger.info(f"IntveEngine Initialized:")
        self.logger.info(f"  Type: {self.df_type}")
        self.logger.info(f"  Data size: {len(self.original_data)}")
        self.logger.info(f"  Bounds: [{self.LB:.6f}, {self.UB:.6f}]")
        self.logger.info(f"  Original Z0: {self.z0:.6f}")
        self.logger.info(f"  Search points per direction: {self.n_points_per_direction}")
        self.logger.info(f"  Dense zone: {self.dense_zone_fraction:.1%} of range")
        self.logger.info(f"  Extrema search tolerance: {self.extrema_search_tolerance}")

    def fit(self, plot: bool = False, update_df_params: bool = True) -> 'IntveEngine':
        """
        Perform interval estimation with improved extrema detection.
        
        Parameters:
        -----------
        plot : bool
            Whether to plot results after fitting
        update_df_params : bool
            Whether to update the original DF object's params with interval results
            
        Returns:
        --------
        self : IntveEngine
            Fitted engine instance
        """
        self.logger.info("Starting Z0-based interval estimation with ordering constraint...")
        
        if self.verbose:
            self.logger.info(f"\nStarting Z0-based interval estimation with ordering constraint...")

        # Record start time
        import time
        start_time = time.time()
        
        try:
            # Reset results
            self._reset_results()
            self.search_results = {
                'lower': {'datum_values': [], 'z0_values': [], 'success_flags': []},
                'upper': {'datum_values': [], 'z0_values': [], 'success_flags': []}
            }
            
            # Test extension capability first
            self.logger.info("Testing data extension capability...")
            self._test_extension_capability()
            
            # Search lower interval: Z0 → LB
            self.logger.info(f"Searching lower interval (Z0 → LB)...")
            self._search_interval('lower')

            # Search upper interval: Z0 → UB
            self.logger.info(f"Searching upper interval (Z0 → UB)...")
            self._search_interval('upper')
            
            # Analyze results with improved extrema detection
            self.logger.info("Analyzing search results and extracting intervals with ordering constraint...")
            self._analyze_and_extract_intervals_with_ordering()
            
            # Record end time and update status
            end_time = time.time()
            self._fitted = True
            
            # Update params with results and statistics
            self.logger.info("Updating parameters with results and statistics...")
            self._update_params_with_results(end_time - start_time)
            
            # Update original DF object params if requested
            if update_df_params:
                self.logger.info("Updating original DF object parameters with interval results...")
                self._update_df_object_params()
            
            if self.verbose:
                self.logger.info("Interval estimation completed successfully.")
                self._print_results()
            
            if plot:
                self.logger.info("Plotting results...")
                self.plot()
                
            return self
            
        except Exception as e:
            error_msg = f"Interval estimation failed: {str(e)}"
            if self.verbose:
                self.logger.error(error_msg)
                self._print_debug_info()
            raise RuntimeError(error_msg) from e
    
    def _search_interval(self, direction: str):
        """
        Search interval in specified direction.
        
        Parameters:
        -----------
        direction : str
            'lower' for Z0→LB search, 'upper' for Z0→UB search
        """
        self.logger.info(f"Searching interval in {direction} direction.")
        # Generate search points for this direction
        search_points = self._generate_search_points(direction)
        
        if len(search_points) == 0:
            self.logger.info(f"  No valid search points for {direction} direction")
            return

        bound_str = "LB" if direction == 'lower' else "UB"
        bound_val = self.LB if direction == 'lower' else self.UB
        self.logger.info(f"  Generated {len(search_points)} points toward {bound_str} ({bound_val:.6f})")

        # Search each point
        self.logger.info(f"  Starting search in {direction} direction...")
        successful_fits = 0
        for i, datum in enumerate(search_points):
            
            try:
                # Compute Z0 with extended datum
                z0_new = self._compute_z0_with_extended_datum(datum)
                
                # Store successful result
                self.search_results[direction]['datum_values'].append(datum)
                self.search_results[direction]['z0_values'].append(z0_new)
                self.search_results[direction]['success_flags'].append(True)
                successful_fits += 1
                
                # Progress reporting
                if self.verbose and (i + 1) % max(1, len(search_points) // 5) == 0:
                    progress = ((i + 1) / len(search_points)) * 100
                    self.logger.info(f"    Progress: {progress:.1f}% | Datum: {datum:.6f} | Z0: {z0_new:.6f}")
                
                # Check for early convergence
                if self._check_convergence(direction) and i >= self.min_search_points:
                    self.logger.info(f"    Early convergence detected after {i+1} points")
                    break
                    
            except Exception as e:
                # Try simple approach for failed cases
                self.logger.warning(f"    Failed at datum {datum:.6f}: {str(e)}. Trying simple approach...")
                try:
                    # Compute Z0 with simple extended datum
                    self.logger.info(f"    Trying simple approach for datum {datum:.6f}")
                    z0_new = self._compute_z0_with_extended_datum_simple(datum)
                    
                    # Store successful result
                    self.search_results[direction]['datum_values'].append(datum)
                    self.search_results[direction]['z0_values'].append(z0_new)
                    self.search_results[direction]['success_flags'].append(True)
                    successful_fits += 1
                    
                except Exception as e2:
                    # Store failed result
                    self.search_results[direction]['datum_values'].append(datum)
                    self.search_results[direction]['z0_values'].append(np.nan)
                    self.search_results[direction]['success_flags'].append(False)
                    
                    if self.verbose and i < 3:  # Show first few errors
                        self.logger.warning(f"    Failed at datum {datum:.6f}: {str(e2)}")
        
        if self.verbose:
            self.logger.info(f"  {direction.capitalize()} search completed: {successful_fits}/{len(search_points)} successful")
    
    def _analyze_and_extract_intervals_with_ordering(self):
        """
        Analyze search results and extract interval parameters with ordering constraint.
        
        Ensures that: ZL < Z0L < Z0 < Z0U < ZU
        If initial extrema don't satisfy this, search for valid alternatives.
        """
        self.logger.info("Analyzing search results with ordering constraint...")
        
        # Collect all successful results
        all_datum_values = []
        all_z0_values = []
        
        for direction in ['lower', 'upper']:
            data = self.search_results[direction]
            for datum, z0, success in zip(data['datum_values'], data['z0_values'], data['success_flags']):
                if success and not np.isnan(z0):
                    all_datum_values.append(datum)
                    all_z0_values.append(z0)
        
        if len(all_z0_values) == 0:
            self.logger.error("No successful fits found. Cannot determine intervals.")
            raise RuntimeError("No successful fits found. Cannot determine intervals.")
        
        all_datum_values = np.array(all_datum_values)
        all_z0_values = np.array(all_z0_values)
        
        if self.verbose:
            self.logger.info(f"  Valid results: {len(all_z0_values)}")
            self.logger.info(f"  Z0 range: [{np.min(all_z0_values):.6f}, {np.max(all_z0_values):.6f}]")
            self.logger.info(f"  Datum range: [{np.min(all_datum_values):.6f}, {np.max(all_datum_values):.6f}]")

        # Find initial extrema
        min_z0_idx = np.argmin(all_z0_values)
        max_z0_idx = np.argmax(all_z0_values)
        
        initial_zl = float(all_datum_values[min_z0_idx])
        initial_z0l = float(all_z0_values[min_z0_idx])
        initial_zu = float(all_datum_values[max_z0_idx])
        initial_z0u = float(all_z0_values[max_z0_idx])
        
        if self.verbose:
            self.logger.info(f"  Initial extrema:")
            self.logger.info(f"    ZL = {initial_zl:.6f}, Z0L = {initial_z0l:.6f}")
            self.logger.info(f"    ZU = {initial_zu:.6f}, Z0U = {initial_z0u:.6f}")

        # Check ordering constraint: ZL < Z0L < Z0 < Z0U < ZU
        ordering_valid = (initial_zl < initial_z0l < self.z0 < initial_z0u < initial_zu)
        
        if ordering_valid:
            if self.verbose:
                self.logger.info(f"  ✓ Ordering constraint satisfied: ZL < Z0L < Z0 < Z0U < ZU")
            
            self.zl = initial_zl
            self.z0l = initial_z0l
            self.zu = initial_zu
            self.z0u = initial_z0u
            
        else:
            if self.verbose:
                self.logger.info(f"  ✗ Ordering constraint violated. Searching for valid extrema...")
                self.logger.info(f"    Current: {initial_zl:.6f} < {initial_z0l:.6f} < {self.z0:.6f} < {initial_z0u:.6f} < {initial_zu:.6f}")

            # Find valid extrema that satisfy ordering constraint
            valid_extrema = self._find_valid_extrema_with_ordering(all_datum_values, all_z0_values)
            
            if valid_extrema is None:
                # Fallback: use best available extrema with warning
                if self.verbose:
                    self.logger.warning(f"  ⚠ No valid extrema found satisfying ordering constraint. Using best available.")
                
                self.zl = initial_zl
                self.z0l = initial_z0l
                self.zu = initial_zu
                self.z0u = initial_z0u
            else:
                self.zl, self.z0l, self.zu, self.z0u = valid_extrema
                if self.verbose:
                    self.logger.info(f"  ✓ Found valid extrema:")
                    self.logger.info(f"    ZL = {self.zl:.6f}, Z0L = {self.z0l:.6f}")
                    self.logger.info(f"    ZU = {self.zu:.6f}, Z0U = {self.z0u:.6f}")
        
        # Compute interval widths
        self.typical_data_interval = self.zu - self.zl
        self.tolerance_interval = self.z0u - self.z0l
        
        # Final validation
        final_ordering_valid = (self.zl < self.z0l < self.z0 < self.z0u < self.zu)
        
        if self.verbose:
            self.logger.info(f"  Final ordering check: {'✓ VALID' if final_ordering_valid else '✗ INVALID'}")
            self.logger.info(f"  Critical points:")
            self.logger.info(f"    ZL = {self.zl:.6f}, Z0L = {self.z0l:.6f}")
            self.logger.info(f"    Z0 = {self.z0:.6f}")
            self.logger.info(f"    Z0U = {self.z0u:.6f}, ZU = {self.zu:.6f}")

    def _find_valid_extrema_with_ordering(self, datum_values: np.ndarray, z0_values: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        """
        Find extrema that satisfy the ordering constraint: ZL < Z0L < Z0 < Z0U < ZU
        
        Parameters:
        -----------
        datum_values : np.ndarray
            Array of datum values
        z0_values : np.ndarray
            Array of corresponding Z0 values
            
        Returns:
        --------
        Optional[Tuple[float, float, float, float]]
            Valid (zl, z0l, zu, z0u) or None if not found
        """
        self.logger.info("Searching for valid extrema satisfying ordering constraint...")

        # Separate lower and upper search results
        lower_mask = datum_values < self.z0
        upper_mask = datum_values > self.z0
        
        lower_datum = datum_values[lower_mask]
        lower_z0 = z0_values[lower_mask]
        upper_datum = datum_values[upper_mask]
        upper_z0 = z0_values[upper_mask]
        
        if len(lower_datum) == 0 or len(upper_datum) == 0:
            self.logger.warning(f"    ✗ Insufficient data for both sides of Z0")
            return None
        
        # Find multiple minima and maxima candidates
        lower_sorted_idx = np.argsort(lower_z0)
        upper_sorted_idx = np.argsort(upper_z0)
        
        # Try different combinations of extrema
        n_candidates = min(5, len(lower_sorted_idx), len(upper_sorted_idx))
        
        for i in range(n_candidates):
            for j in range(n_candidates):
                # Try i-th minimum and j-th maximum
                min_idx = lower_sorted_idx[i]
                max_idx = upper_sorted_idx[-(j+1)]  # j-th from the end (highest)
                
                candidate_zl = float(lower_datum[min_idx])
                candidate_z0l = float(lower_z0[min_idx])
                candidate_zu = float(upper_datum[max_idx])
                candidate_z0u = float(upper_z0[max_idx])
                
                # Check ordering constraint
                if (candidate_zl < candidate_z0l < self.z0 < candidate_z0u < candidate_zu):
                    self.logger.info(f"    ✓ Found valid combination (min_rank={i+1}, max_rank={j+1})")
                    self.logger.info(f"      {candidate_zl:.6f} < {candidate_z0l:.6f} < {self.z0:.6f} < {candidate_z0u:.6f} < {candidate_zu:.6f}")

                    return (candidate_zl, candidate_z0l, candidate_zu, candidate_z0u)
        
        # If no valid combination found, try relaxed search
        self.logger.warning(f"    No strict extrema found. Trying relaxed search...")
        
        return self._find_extrema_relaxed_search(datum_values, z0_values)
    
    def _find_extrema_relaxed_search(self, datum_values: np.ndarray, z0_values: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        """
        Relaxed search for extrema when strict extrema don't satisfy ordering.
        
        Parameters:
        -----------
        datum_values : np.ndarray
            Array of datum values
        z0_values : np.ndarray
            Array of corresponding Z0 values
            
        Returns:
        --------
        Optional[Tuple[float, float, float, float]]
            Valid (zl, z0l, zu, z0u) or None if not found
        """
        self.logger.info("Performing relaxed search for extrema...")
        # Split data into lower and upper regions
        lower_mask = datum_values < self.z0
        upper_mask = datum_values > self.z0
        
        lower_datum = datum_values[lower_mask]
        lower_z0 = z0_values[lower_mask]
        upper_datum = datum_values[upper_mask]
        upper_z0 = z0_values[upper_mask]
        
        if len(lower_datum) == 0 or len(upper_datum) == 0:
            return None
        
        # For lower region: find datum that gives Z0 < Z0_original and ZL < Z0L
        valid_lower_mask = (lower_z0 < self.z0) & (lower_datum < lower_z0)
        if np.any(valid_lower_mask):
            valid_lower_datum = lower_datum[valid_lower_mask]
            valid_lower_z0 = lower_z0[valid_lower_mask]
            
            # Choose the one with minimum Z0
            min_idx = np.argmin(valid_lower_z0)
            candidate_zl = valid_lower_datum[min_idx]
            candidate_z0l = valid_lower_z0[min_idx]
        else:
            # Fallback: use extrema even if ordering is violated
            min_idx = np.argmin(lower_z0)
            candidate_zl = lower_datum[min_idx]
            candidate_z0l = lower_z0[min_idx]
        
        # For upper region: find datum that gives Z0 > Z0_original and ZU > Z0U
        valid_upper_mask = (upper_z0 > self.z0) & (upper_datum > upper_z0)
        if np.any(valid_upper_mask):
            valid_upper_datum = upper_datum[valid_upper_mask]
            valid_upper_z0 = upper_z0[valid_upper_mask]
            
            # Choose the one with maximum Z0
            max_idx = np.argmax(valid_upper_z0)
            candidate_zu = valid_upper_datum[max_idx]
            candidate_z0u = valid_upper_z0[max_idx]
        else:
            # Fallback: use extrema even if ordering is violated
            max_idx = np.argmax(upper_z0)
            candidate_zu = upper_datum[max_idx]
            candidate_z0u = upper_z0[max_idx]
        
        if self.verbose:
            ordering_check = (candidate_zl < candidate_z0l < self.z0 < candidate_z0u < candidate_zu)
            self.logger.info(f"    Relaxed search result: {'✓ VALID' if ordering_check else '✗ INVALID'}")
            self.logger.info(f"      {candidate_zl:.6f} < {candidate_z0l:.6f} < {self.z0:.6f} < {candidate_z0u:.6f} < {candidate_zu:.6f}")

        return (candidate_zl, candidate_z0l, candidate_zu, candidate_z0u)
    
    def _generate_search_points(self, direction: str) -> np.ndarray:
        """Generate search points for given direction with dense sampling near Z0."""
        self.logger.info(f"Generating search points in the {direction} direction...")
        
        if direction == 'lower':
            self.logger.info("  Generating points toward LB...")
            # Search from Z0 toward LB
            full_range = self.z0 - self.LB
            if full_range <= 0:
                return np.array([])
            
            # Apply safety margin
            margin = full_range * self.boundary_margin_factor
            search_start = self.z0
            search_end = self.LB + margin
            
            if search_start <= search_end:
                return np.array([])
            
            # Dense zone: near Z0
            dense_range = full_range * self.dense_zone_fraction
            dense_start = max(search_end, search_start - dense_range)
            
            # Generate points
            n_dense = int(self.n_points_per_direction * self.dense_points_fraction)
            n_sparse = self.n_points_per_direction - n_dense
            
            # Dense points (linear spacing)
            dense_points = np.linspace(search_start, dense_start, n_dense + 1)[1:]  # Exclude Z0
            
            # Sparse points (logarithmic spacing toward boundary)
            if n_sparse > 0 and dense_start > search_end:
                # Logarithmic ratios for smooth transition
                log_space = np.logspace(0, 2, n_sparse + 1)[1:]  # [10^0, 10^2] range
                log_ratios = (log_space - 1) / (100 - 1)  # Normalize to [0, 1]
                sparse_points = search_end + log_ratios * (dense_start - search_end)
            else:
                sparse_points = np.array([])
            
            # Combine and sort (Z0 → LB direction)
            all_points = np.concatenate([dense_points, sparse_points])
            return np.sort(all_points)[::-1]  # Descending order
            
        else:  # upper direction
            self.logger.info("  Generating points toward UB...")
            # Search from Z0 toward UB
            full_range = self.UB - self.z0
            if full_range <= 0:
                return np.array([])
            
            # Apply safety margin
            margin = full_range * self.boundary_margin_factor
            search_start = self.z0
            search_end = self.UB - margin
            
            if search_start >= search_end:
                return np.array([])
            
            # Dense zone: near Z0
            dense_range = full_range * self.dense_zone_fraction
            dense_end = min(search_end, search_start + dense_range)
            
            # Generate points
            n_dense = int(self.n_points_per_direction * self.dense_points_fraction)
            n_sparse = self.n_points_per_direction - n_dense
            
            # Dense points (linear spacing)
            dense_points = np.linspace(search_start, dense_end, n_dense + 1)[1:]  # Exclude Z0
            
            # Sparse points (logarithmic spacing toward boundary)
            if n_sparse > 0 and dense_end < search_end:
                # Logarithmic ratios for smooth transition
                log_space = np.logspace(0, 2, n_sparse + 1)[1:]  # [10^0, 10^2] range
                log_ratios = (log_space - 1) / (100 - 1)  # Normalize to [0, 1]
                sparse_points = dense_end + log_ratios * (search_end - dense_end)
            else:
                sparse_points = np.array([])
            
            # Combine and sort (Z0 → UB direction)
            all_points = np.concatenate([dense_points, sparse_points])
            return np.sort(all_points)  # Ascending order
    
    def _check_convergence(self, direction: str) -> bool:
        """Check if Z0 values have converged in recent window."""

        self.logger.info(f"  Checking convergence in {direction} direction...")
        
        z0_values = self.search_results[direction]['z0_values']
        success_flags = self.search_results[direction]['success_flags']
        
        # Get successful Z0 values
        valid_z0 = [z0 for z0, success in zip(z0_values, success_flags) 
                   if success and not np.isnan(z0)]
        
        if len(valid_z0) < self.convergence_window:
            return False
        
        # Check recent window for convergence
        recent_z0 = valid_z0[-self.convergence_window:]
        z0_std = np.std(recent_z0)
        
        return z0_std < self.convergence_threshold
    
    def _compute_z0_with_extended_datum(self, datum: float) -> float:
        """Compute Z0 for data extended with given datum."""

        self.logger.info(f"  Computing Z0 with extended datum: {datum:.6f}")    
        
        # Create extended data
        extended_data = np.append(self.original_data, datum)
        
        # Handle weights if present
        extended_weights = None
        if self.weights is not None:
            extended_weights = np.append(self.weights, 1.0)
        
        # Create new DF instance with extended data
        df_extended = self._create_df_instance(extended_data, extended_weights)
        
        # Fit and extract Z0
        df_extended.fit(plot=False)
        
        if not hasattr(df_extended, 'z0') or df_extended.z0 is None:
            self.logger.error("Z0 not computed for extended DF")
            raise ValueError("Z0 not computed for extended DF")
        
        return float(df_extended.z0)
    
    def _compute_z0_with_extended_datum_simple(self, datum: float) -> float:
        """Compute Z0 with minimal parameters (fallback method)."""

        self.logger.info(f"  (Simple) Computing Z0 with extended datum: {datum:.6f}")
        
        # Create extended data
        extended_data = np.append(self.original_data, datum)
        
        # Use minimal parameters
        minimal_params = {
            'data': extended_data,
            'LB': self.LB,
            'UB': self.UB,
            'n_points': min(200, len(extended_data) * 3),
            'verbose': False,
            'tolerance': self.tolerance * 10,  # Relaxed tolerance
            'catch': False,
            'flush': False
        }
        
        # Create DF with minimal parameters
        if self.df_type == 'EGDF':
            df_extended = EGDF(**minimal_params)
        else:  # ELDF
            df_extended = ELDF(**minimal_params)
        
        # Fit and extract Z0
        df_extended.fit(plot=False)
        
        if not hasattr(df_extended, 'z0') or df_extended.z0 is None:
            self.logger.error("Z0 not computed for extended DF (simple)")
            raise ValueError("Z0 not computed for extended DF (simple)")
        
        return float(df_extended.z0)
    
    def _create_df_instance(self, data: np.ndarray, weights: Optional[np.ndarray] = None):
        """Create DF instance with given data using original parameters."""
        
        self.logger.info("Creating DF instance with extended data...")  

        # Use adaptive n_points for efficiency
        n_points = min(400, len(data) * 4)
        
        # Relaxed tolerance for extended DF fitting
        extended_tolerance = self.tolerance * 5
        
        # Build parameters carefully
        common_params = {
            'data': data,
            'LB': self.LB,
            'UB': self.UB,
            'tolerance': extended_tolerance,
            'data_form': self.data_form,
            'n_points': n_points,
            'homogeneous': self.homogeneous,
            'verbose': False,
            'max_data_size': self.max_data_size,
            'catch': False,
            'flush': False
        }
        
        # Only add DLB/DUB if they are valid
        if self.DLB is not None:
            common_params['DLB'] = self.DLB
        if self.DUB is not None:
            common_params['DUB'] = self.DUB
            
        # Only add weights if provided
        if weights is not None:
            common_params['weights'] = weights
        
        if self.df_type == 'EGDF':
            return EGDF(
                S=self.S_opt,
                wedf=self.wedf,
                opt_method=self.opt_method,
                **common_params
            )
        else:  # ELDF
            return ELDF(**common_params)
    
    def _test_extension_capability(self):
        """Test if data can be extended successfully."""
        self.logger.info("Testing data extension capability...")
        
        # Try a small extension near Z0
        test_datum = self.z0 + 0.01 * (self.UB - self.z0)
        
        try:
            test_z0 = self._compute_z0_with_extended_datum(test_datum)
            self.logger.info(f"  Extension test successful: Z0_new = {test_z0:.6f}")
        except Exception as e:
            self.logger.error(f"  First extension test failed: {str(e)}")
            self.logger.info(f"  First extension test failed, trying simpler approach...")

            # Try with minimal parameters
            try:
                test_z0 = self._compute_z0_with_extended_datum_simple(test_datum)
                self.logger.info(f"  Simple extension test successful: Z0_new = {test_z0:.6f}")
            except Exception as e2:
                self.logger.error(f"  Simple extension test failed: {str(e2)}")
                raise RuntimeError(f"Cannot extend data: {str(e2)}")
    
    def _update_params_with_results(self, fit_time: float):
        """Update params dictionary with fitting results and statistics."""

        self.logger.info("Updating parameters with fitting results and statistics...")
        
        # Search statistics
        lower_success = sum(self.search_results['lower']['success_flags'])
        upper_success = sum(self.search_results['upper']['success_flags'])
        lower_total = len(self.search_results['lower']['datum_values'])
        upper_total = len(self.search_results['upper']['datum_values'])
        total_success = lower_success + upper_success
        total_attempts = lower_total + upper_total
        
        # Ordering validation
        ordering_valid = (self.zl < self.z0l < self.z0 < self.z0u < self.zu)

        # self z0
        self.z0 = self.z0
        
        # Update params with complete results
        self.params.update({
            # Core interval results
            'ZL': float(self.zl),
            'Z0L': float(self.z0l),
            'Z0': float(self.z0),
            'Z0U': float(self.z0u),
            'ZU': float(self.zu),

            # Interval measures
            'tolerance_interval': [float(self.z0l), float(self.z0u)],
            'typical_data_interval': [float(self.zl), float(self.zu)],
            'tolerance_interval_width': float(self.tolerance_interval),
            'typical_data_interval_width': float(self.typical_data_interval),
            
            # Ordering validation
            'ordering_validation': {
                'constraint_satisfied': ordering_valid,
                'constraint_formula': 'ZL < Z0L < Z0 < Z0U < ZU',
                'values': [self.zl, self.z0l, self.z0, self.z0u, self.zu],
                'differences': [
                    self.z0l - self.zl,
                    self.z0 - self.z0l,
                    self.z0u - self.z0,
                    self.zu - self.z0u
                ]
            },
            
            # Relative measures
            'tolerance_to_bounds_ratio': self.tolerance_interval / (self.UB - self.LB),
            'typical_to_bounds_ratio': self.typical_data_interval / (self.UB - self.LB),
            'typical_to_tolerance_ratio': self.typical_data_interval / self.tolerance_interval if self.tolerance_interval > 0 else np.inf,
            
            # Data coverage
            'data_within_tolerance': self._count_data_in_interval(self.z0l, self.z0u),
            'data_within_typical': self._count_data_in_interval(self.zl, self.zu),
            'data_within_tolerance_fraction': self._count_data_in_interval(self.z0l, self.z0u) / len(self.original_data),
            'data_within_typical_fraction': self._count_data_in_interval(self.zl, self.zu) / len(self.original_data),
            
            # Search statistics
            'search_statistics': {
                'total_attempts': total_attempts,
                'total_successful': total_success,
                'success_rate': total_success / total_attempts if total_attempts > 0 else 0,
                'lower_search': {
                    'attempts': lower_total,
                    'successful': lower_success,
                    'success_rate': lower_success / lower_total if lower_total > 0 else 0
                },
                'upper_search': {
                    'attempts': upper_total,
                    'successful': upper_success,
                    'success_rate': upper_success / upper_total if upper_total > 0 else 0
                }
            },
            
            # Timing and status
            'fitted': True,
            'fit_time': fit_time,
            'fit_timestamp': np.datetime64('now'),
            
            # Quality metrics
            'interval_quality': {
                'z0_variation_range': self.z0u - self.z0l,
                'datum_variation_range': self.zu - self.zl,
                'z0_stability': 1.0 - (self.z0u - self.z0l) / abs(self.z0) if self.z0 != 0 else 1.0,
                'interval_symmetry': abs((self.zu + self.zl) / 2 - self.z0) / (self.UB - self.LB),
                'ordering_constraint_satisfied': ordering_valid
            }
        })
    
    def _count_data_in_interval(self, lower: float, upper: float) -> int:
        """Count how many data points fall within the given interval."""
        self.logger.info(f"Counting data points in interval [{lower:.6f}, {upper:.6f}]...")
        return np.sum((self.original_data >= lower) & (self.original_data <= upper))
    
    def _update_df_object_params(self):
        """Update the original DF object's params with interval results."""
        
        self.lower.info("Updating original DF object parameters with interval results...")

        if not hasattr(self.df_object, 'params'):
            self.df_object.params = {}
        
        # Create interval-specific parameter dictionary
        interval_params = {
            'interval_estimation': {
                # Core results
                'ZL': self.zl,
                'Z0L': self.z0l,
                'Z0U': self.z0u,
                'ZU': self.zu,
                'tolerance_interval': [self.z0l, self.z0u],
                'typical_data_interval': [self.zl, self.zu],
                'tolerance_interval_width': self.tolerance_interval,
                'typical_data_interval_width': self.typical_data_interval,
                
                # Ordering validation
                'ordering_constraint_satisfied': (self.zl < self.z0l < self.z0 < self.z0u < self.zu),
                
                # Summary statistics
                'data_coverage': {
                    'tolerance_count': self._count_data_in_interval(self.z0l, self.z0u),
                    'typical_count': self._count_data_in_interval(self.zl, self.zu),
                    'tolerance_fraction': self._count_data_in_interval(self.z0l, self.z0u) / len(self.original_data),
                    'typical_fraction': self._count_data_in_interval(self.zl, self.zu) / len(self.original_data),
                },
                
                # Method information
                'method': 'Z0-based interval estimation with ordering constraint',
                'engine_type': 'IntveEngine',
                'search_points_per_direction': self.n_points_per_direction,
                'successful_fits': sum(self.search_results['lower']['success_flags']) + sum(self.search_results['upper']['success_flags']),
                'fit_timestamp': str(np.datetime64('now'))
            }
        }
        
        # Update DF object params
        self.df_object.params.update(interval_params)
        
        self.logger.info(f"Updated {self.df_type} object params with interval estimation results")
    
    def _print_results(self):
        """Print formatted results."""
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"Z0-BASED INTERVAL ESTIMATION RESULTS - ({self.df_type})")
        self.logger.info(f"{'='*70}")
        
        self.logger.info(f"Original Configuration:")
        self.logger.info(f"  Data size: {len(self.original_data)}")
        self.logger.info(f"  Bounds: [{self.LB:.6f}, {self.UB:.6f}]")
        self.logger.info(f"  Original Z0: {self.z0:.6f}")
        self.logger.info(f"")
        
        # Ordering constraint check
        ordering_valid = (self.zl < self.z0l < self.z0 < self.z0u < self.zu)
        self.logger.info(f"Ordering Constraint: ZL < Z0L < Z0 < Z0U < ZU")
        self.logger.info(f"  Status: {'✓ SATISFIED' if ordering_valid else '✗ VIOLATED'}")
        self.logger.info(f"  Values: {self.zl:.6f} < {self.z0l:.6f} < {self.z0:.6f} < {self.z0u:.6f} < {self.zu:.6f}")

        if not ordering_valid:
            self.logger.info(f"  ⚠ Warning: Ordering constraint not satisfied. Results may be suboptimal.")
        self.logger.info(f"")
        
        self.logger.info(f"Critical Points:")
        self.logger.info(f"  ZL  (datum for min Z0):  {self.zl:.6f}")
        self.logger.info(f"  Z0L (minimum Z0):        {self.z0l:.6f}")
        self.logger.info(f"  Z0  (original):          {self.z0:.6f}")
        self.logger.info(f"  Z0U (maximum Z0):        {self.z0u:.6f}")
        self.logger.info(f"  ZU  (datum for max Z0):  {self.zu:.6f}")
        self.logger.info(f"")
        
        self.logger.info(f"Intervals:")
        self.logger.info(f"  Typical Data:  [{self.zl:.6f}, {self.zu:.6f}]  (width: {self.typical_data_interval:.6f})")
        self.logger.info(f"  Tolerance:     [{self.z0l:.6f}, {self.z0u:.6f}]  (width: {self.tolerance_interval:.6f})")
        self.logger.info(f"")
        
        # Data coverage
        tol_count = self._count_data_in_interval(self.z0l, self.z0u)
        typ_count = self._count_data_in_interval(self.zl, self.zu)
        total_data = len(self.original_data)

        self.logger.info(f"Data Coverage:")
        self.logger.info(f"  Within tolerance interval: {tol_count}/{total_data} ({tol_count/total_data:.1%})")
        self.logger.info(f"  Within typical data interval: {typ_count}/{total_data} ({typ_count/total_data:.1%})")
        self.logger.info(f"")

        # Search summary
        lower_success = sum(self.search_results['lower']['success_flags'])
        upper_success = sum(self.search_results['upper']['success_flags'])
        lower_total = len(self.search_results['lower']['datum_values'])
        upper_total = len(self.search_results['upper']['datum_values'])

        self.logger.info(f"Search Summary:")
        self.logger.info(f"  Lower direction: {lower_success}/{lower_total} successful")
        self.logger.info(f"  Upper direction: {upper_success}/{upper_total} successful")
        self.logger.info(f"  Total valid fits: {lower_success + upper_success}")
        self.logger.info(f"  Fit time: {self.params.get('fit_time', 0):.3f} seconds")
        self.logger.info(f"{'='*70}")
    
    def _print_debug_info(self):
        """Print debug information when fitting fails."""
        self.logger.info(' Fitting failed or produced invalid results. Debug information:')
        self.logger.info(f"Original data: {self.original_data}")
        self.logger.info(f"Data stats: mean={np.mean(self.original_data):.6f}, std={np.std(self.original_data):.6f}")
        self.logger.info(f"Bounds: LB={self.LB:.6f}, UB={self.UB:.6f}")
        self.logger.info(f"Z0: {self.z0:.6f}")
        self.logger.info(f"DLB: {self.DLB}, DUB: {self.DUB}")
        
        # Show search results summary
        for direction in ['lower', 'upper']:
            data = self.search_results[direction]
            if len(data['datum_values']) > 0:
                success_count = sum(data['success_flags'])
                total_count = len(data['datum_values'])
                self.logger.info(f"{direction.capitalize()} search: {success_count}/{total_count} successful")
    
    def get_intervals(self, decimals: int = 6) -> Dict[str, float]:
        """Get interval results as dictionary."""
        self.logger.info("Retrieving interval results as dictionary...")
        
        if not self._fitted:
            raise RuntimeError("Must fit before getting intervals")
        
        return {
            'ZL': round(self.zl, decimals),
            'Z0L': round(self.z0l, decimals), 
            'Z0': round(self.z0, decimals),
            'Z0U': round(self.z0u, decimals),
            'ZU': round(self.zu, decimals),
            'typical_data_interval': round(self.typical_data_interval, decimals),
            'tolerance_interval': round(self.tolerance_interval, decimals),
            'LB': round(self.LB, decimals),
            'UB': round(self.UB, decimals),
            'ordering_constraint_satisfied': (self.zl < self.z0l < self.z0 < self.z0u < self.zu)
        }
    
    def plot(self, figsize: Tuple[int, int] = (12, 8)):
        """Plot interval estimation results."""
        self.logger.info("Plotting interval estimation results...")
        
        if not self._fitted:
            self.logger.error("Must fit before plotting")
            raise RuntimeError("Must fit before plotting")
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.error("matplotlib required for plotting")
            raise ImportError("matplotlib required for plotting")
        
        # Create main Z0 variation plot
        self._plot_z0_variation(figsize)
        
        # Create distribution plot if requested
        # if plot_distribution:
        #     self._plot_distribution_with_intervals(figsize, eldf_plot)
    
    def _plot_z0_variation(self, figsize: Tuple[int, int] = (12, 8)):
        """Plot Z0 variation with improved legend and ordering validation."""
        self.logger.info("Plotting Z0 variation...")
        import matplotlib.pyplot as plt
        
        # Collect valid data points
        datum_vals, z0_vals, colors = [], [], []
        
        # Lower search (blue)
        lower_data = self.search_results['lower']
        for datum, z0, success in zip(lower_data['datum_values'], lower_data['z0_values'], lower_data['success_flags']):
            if success and not np.isnan(z0):
                datum_vals.append(datum)
                z0_vals.append(z0)
                colors.append('blue')
        
        # Upper search (red)  
        upper_data = self.search_results['upper']
        for datum, z0, success in zip(upper_data['datum_values'], upper_data['z0_values'], upper_data['success_flags']):
            if success and not np.isnan(z0):
                datum_vals.append(datum)
                z0_vals.append(z0)
                colors.append('red')
        
        if len(datum_vals) == 0:
            self.logger.info("No valid data for plotting")
            return
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Scatter points by search direction
        datum_vals = np.array(datum_vals)
        z0_vals = np.array(z0_vals)
        colors = np.array(colors)
        
        blue_mask = colors == 'blue'
        red_mask = colors == 'red'
        
        if np.any(blue_mask):
            ax.scatter(datum_vals[blue_mask], z0_vals[blue_mask], 
                      c='blue', alpha=0.6, s=20, label='Lower Search (Z0→LB)')
        if np.any(red_mask):
            ax.scatter(datum_vals[red_mask], z0_vals[red_mask], 
                      c='red', alpha=0.6, s=20, label='Upper Search (Z0→UB)')
        
        # Smooth curve if enough points
        if len(datum_vals) > 20:
            sort_idx = np.argsort(datum_vals)
            ax.plot(datum_vals[sort_idx], z0_vals[sort_idx], 'k-', 
                   alpha=0.4, linewidth=1.5, label='Z0 Variation Curve')
        
        # Critical points
        ax.scatter([self.zl], [self.z0l], marker='v', s=150, color='purple',
                  edgecolor='black', linewidth=2, zorder=10, 
                  label=f'ZL,Z0L ({self.zl:.4f},{self.z0l:.4f})')
        ax.scatter([self.z0], [self.z0], marker='s', s=150, color='green',
                  edgecolor='black', linewidth=2, zorder=10, 
                  label=f'Z0 ({self.z0:.4f})')
        ax.scatter([self.zu], [self.z0u], marker='^', s=150, color='orange',
                  edgecolor='black', linewidth=2, zorder=10, 
                  label=f'Z0U, ZU ({self.z0u:.4f},{self.zu:.4f})')

        # Reference lines
        ax.axvline(x=self.zl, color='purple', linestyle='--', alpha=0.7, linewidth=1)
        ax.axvline(x=self.z0, color='green', linestyle='-', alpha=0.8, linewidth=2)
        ax.axvline(x=self.zu, color='orange', linestyle='--', alpha=0.7, linewidth=1)
        ax.axhline(y=self.z0l, color='purple', linestyle=':', alpha=0.7, linewidth=1)
        ax.axhline(y=self.z0u, color='orange', linestyle=':', alpha=0.7, linewidth=1)
        
        # Add interval information and ordering status to legend
        ordering_valid = (self.zl < self.z0l < self.z0 < self.z0u < self.zu)
        ordering_status = "✓ VALID" if ordering_valid else "✗ INVALID"
        
        tol_interval_str = f"Tolerance Interval: [{self.z0l:.4f}, {self.z0u:.4f}]"
        typ_interval_str = f"Typical Data Interval: [{self.zl:.4f}, {self.zu:.4f}]"
        ordering_str = f"Ordering Constraint: {ordering_status}"
        
        # Create invisible plot points for legend entries
        ax.plot([], [], ' ', label=tol_interval_str, color='lightgreen', alpha=0.7)
        ax.plot([], [], ' ', label=typ_interval_str, color='lightblue', alpha=0.7)
        ax.plot([], [], ' ', label=ordering_str, color='red' if not ordering_valid else 'green', alpha=0.7)
        
        # Labels and formatting
        ax.set_xlabel('Datum Value', fontsize=12, fontweight='bold')
        ax.set_ylabel('Z0 Value', fontsize=12, fontweight='bold')
        
        title = f'Z0-Based Interval Estimation ({self.df_type})'
        if not ordering_valid:
            title += ' - ⚠ Ordering Constraint Violated'
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        if self.verbose:
            # Print summary
            self.logger.info(f"\nZ0 Variation Plot Summary:")
            self.logger.info(f"  Total valid points: {len(datum_vals)}")
            self.logger.info(f"  Typical data interval: [{self.zl:.6f}, {self.zu:.6f}] (width: {self.typical_data_interval:.6f})")
            self.logger.info(f"  Tolerance interval: [{self.z0l:.6f}, {self.z0u:.6f}] (width: {self.tolerance_interval:.6f})")
            self.logger.info(f"  Ordering constraint: {'✓ SATISFIED' if ordering_valid else '✗ VIOLATED'}")

    # def _plot_distribution_with_intervals(self, figsize: Tuple[int, int] = (12, 8), 
    #                                     eldf_plot: bool = True):
    #     """Plot ELDF/PDF distribution with interval markers and filled areas."""
        
    #     import matplotlib.pyplot as plt
        
    #     # Create figure
    #     fig, ax = plt.subplots(1, 1, figsize=figsize)
        
    #     # Get x range for plotting (slightly beyond bounds)
    #     x_margin = (self.UB - self.LB) * 0.05
    #     x_min = self.LB - x_margin
    #     x_max = self.UB + x_margin
    #     x = np.linspace(x_min, x_max, 1000)
        
    #     # Compute and plot ELDF or PDF
    #     if eldf_plot:
    #         try:
    #             y = self.df_object.eldf(x)
    #             ax.plot(x, y, 'k-', linewidth=2, label=f'{self.df_type} Function', alpha=0.8)
    #             y_label = f'{self.df_type} Value'
    #             plot_title = f'{self.df_type} with Intervals'
    #         except Exception as e:
    #             print(f"Could not compute ELDF: {e}")
    #             return
    #     else:
    #         try:
    #             y = self.df_object.pdf(x)
    #             ax.plot(x, y, 'k-', linewidth=2, label='PDF', alpha=0.8)
    #             y_label = 'Probability Density'
    #             plot_title = 'PDF with Intervals'
    #         except Exception as e:
    #             print(f"Could not compute PDF: {e}")
    #             return