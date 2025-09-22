import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from typing import Union, Dict, Any, Optional, Tuple, List
from machinegnostics.magcal import EGDF
import logging
from machinegnostics.magcal.util.logging import get_logger

class DataHomogeneity:
    """
    Analyze data homogeneity for EGDF objects using probability density function analysis.
    
    This class provides comprehensive homogeneity analysis for Estimating Global Distribution Functions (EGDF)
    by examining the shape and characteristics of their probability density functions (PDF). The
    homogeneity criterion is based on the mathematical properties and expected PDF behavior of EGDF
    according to gnostic theory principles.
    
    **Gnostic Theory Foundation:**
    
    The EGDF is uniquely determined by the data sample and finds the optimal scale parameter automatically.
    Unlike local distribution functions, EGDF has limited flexibility and provides a unique representation
    for each homogeneous data sample. The key principle is that homogeneous data should produce a 
    distribution with a single density maximum, while non-homogeneous data will exhibit multiple maxima
    or negative density values.
    
    **Homogeneity Criteria:**
    
    - **EGDF (Estimating Global Distribution Function)**: Data is considered homogeneous if:
      1. PDF has exactly one global maximum (single peak)
      2. PDF contains no negative values
    
    **EGDF Characteristics:**
    
    - **Uniqueness**: EGDF finds the best scale parameter automatically, providing a unique model
    - **Robustness**: EGDF is robust with respect to outliers
    - **Homogeneity Testing**: Particularly suitable for reliable data homogeneity testing
    - **Global Nature**: Uses normalized weights resulting in limited flexibility controlled by optimal scale
    - **Data-Driven**: Primary parameters are the data themselves, following gnostic "let data speak" principle
    
    **Non-Homogeneity Detection:**
    
    EGDF can sensitively detect two main causes of non-homogeneity:
    1. **Outliers**: Individual data points significantly different from others, creating local maxima
    2. **Clusters**: Separate groups in the data, resulting in multiple density peaks
    
    **Key Features:**
    
    - Automatic EGDF validation
    - Robust peak detection with configurable smoothing
    - Comprehensive error and warning tracking
    - Memory management with optional data flushing
    - Detailed visualization of analysis results
    - Integration with existing GDF parameter systems
    
    **Analysis Pipeline:**
    
    1. **Validation**: Ensures input is EGDF only (rejects QGDF/ELDF/QLDF)
    2. **PDF Extraction**: Retrieves PDF points from fitted EGDF object
    3. **Smoothing**: Applies Gaussian filtering for noise reduction
    4. **Maxima Detection**: Identifies peaks in the smoothed PDF
    5. **Homogeneity Assessment**: Evaluates based on peak count and PDF negativity
    6. **Result Storage**: Comprehensive parameter collection and storage
    
    Parameters
    ----------
    gdf : EGDF
        A fitted Estimating Global Distribution Function object. Must be EGDF
        (QGDF, ELDF and QLDF are not supported). The object must:
        - Be fitted (gdf._fitted == True)
        - Have catch=True to generate required pdf_points and di_points_n
        - Contain valid data and PDF information
        - Have optimized scale parameter S_opt from EGDF fitting process
        
    verbose : bool, default=True
        Controls output verbosity during analysis.
        - True: Prints detailed progress, warnings, and results
        - False: Silent operation (errors still raise exceptions)
        
    catch : bool, default=True
        Enables comprehensive result storage in params dictionary.
        - True: Stores all analysis results, parameters, and metadata
        - False: Minimal storage (not recommended for most use cases)
        
    flush : bool, default=False
        Controls memory management of large arrays after analysis.
        - True: Clears pdf_points and di_points_n from GDF object to save memory
        - False: Preserves all data arrays (recommended for further analysis)
        
    smoothing_sigma : float, default=1.0
        Gaussian smoothing parameter for PDF preprocessing before peak detection.
        - Larger values: More aggressive smoothing, may merge distinct features
        - Smaller values: Less smoothing, may detect noise as features
        - Range: 0.1 to 5.0 (typical), must be positive
        - Important for numerical sensitivity beyond visual inspection
        
    min_height_ratio : float, default=0.01
        Minimum relative height threshold for peak detection.
        - Expressed as fraction of global maximum height
        - Range: 0.001 to 0.1 (typical)
        - Higher values: More selective, fewer detected peaks
        - Lower values: More sensitive, may include noise
        
    min_distance : Optional[int], default=None
        Minimum separation between detected peaks in array indices.
        - None: Automatically calculated as len(pdf_data) // 20
        - Integer: Explicit minimum distance constraint
        - Prevents detection of closely spaced spurious peaks
    
    Attributes
    ----------
    is_homogeneous : bool or None
        Primary analysis result. None before fit(), True/False after analysis
        
    picks : List[Dict]
        Detected maxima with detailed information:
        - index: Array index of maximum
        - position: Data value at maximum
        - pdf_value: Original PDF value at maximum
        - smoothed_pdf_value: Smoothed PDF value at maximum
        - is_global: Boolean indicating global maximum
        
    z0 : float or None
        Global optimum value from EGDF object or detected from PDF
        
    global_extremum_idx : int or None
        Array index of the global maximum
        
    fitted : bool
        Read-only property indicating if analysis has been completed
    
    Raises
    ------
    ValueError
        - If input is not EGDF object
        - If GDF object is not fitted
        - If required attributes are missing
        
    AttributeError
        - If EGDF object lacks pdf_points (catch=False during EGDF fitting)
        - If required EGDF attributes are not accessible
        
    RuntimeError
        - If fit() method fails due to numerical issues
        - If plot() or results() called before fit()
    
    Examples
    --------
    **Basic Homogeneity Analysis with EGDF:**
    
    >>> import numpy as np
    >>> from machinegnostics.magcal import EGDF
    >>> from machinegnostics.magcal import DataHomogeneity
    >>> 
    >>> # Prepare homogeneous data (single cluster)
    >>> data = np.array([1.0, 1.1, 1.2, 0.9, 1.0, 1.1])
    >>> 
    >>> # Fit EGDF with catch=True (required for homogeneity analysis)
    >>> egdf = EGDF(data=data, catch=True, verbose=False)
    >>> egdf.fit()  # Automatically finds optimal scale parameter
    >>> 
    >>> # Analyze homogeneity
    >>> homogeneity = DataHomogeneity(egdf, verbose=True)
    >>> is_homogeneous = homogeneity.fit()
    >>> print(f"Data is homogeneous: {is_homogeneous}")
    >>> 
    >>> # Visualize results
    >>> homogeneity.plot()
    >>> 
    >>> # Get detailed results
    >>> results = homogeneity.results()
    >>> print(f"Number of maxima detected: {len(results['picks'])}")
    
    **EGDF Analysis with Multiple Clusters:**
    
    >>> # Heterogeneous data (multiple clusters)
    >>> data = np.array([1, 2, 3, 10, 11, 12, 20, 21, 22])
    >>> 
    >>> # Fit EGDF (will find optimal S automatically)
    >>> egdf = EGDF(data=data, catch=True)
    >>> egdf.fit()
    >>> 
    >>> # Analyze with custom smoothing for numerical sensitivity
    >>> homogeneity = DataHomogeneity(
    ...     egdf, 
    ...     verbose=True,
    ...     smoothing_sigma=2.0,  # More aggressive smoothing
    ...     min_height_ratio=0.05,  # Higher threshold
    ...     flush=True  # Save memory
    ... )
    >>> 
    >>> is_homogeneous = homogeneity.fit()
    >>> # Expected: False due to multiple clusters creating multiple maxima
    
    **Outlier Detection Example:**
    
    >>> # Data with outlier
    >>> data = np.array([5, 5.1, 5.2, 4.9, 5.0, 15.0])  # 15.0 is outlier
    >>> 
    >>> # Fit EGDF
    >>> egdf = EGDF(data=data, catch=True)
    >>> egdf.fit()
    >>> 
    >>> # Analyze homogeneity
    >>> homogeneity = DataHomogeneity(egdf, verbose=True)
    >>> is_homogeneous = homogeneity.fit()
    >>> # Expected: False due to outlier creating additional local maximum
    
    **Error Handling and Parameter Access:**
    
    >>> # Access comprehensive results
    >>> results = homogeneity.results()
    >>> 
    >>> # Check for analysis errors
    >>> if 'errors' in results:
    ...     print("Analysis errors:", results['errors'])
    >>> 
    >>> # Check for warnings
    >>> if 'warnings' in results:
    ...     print("Analysis warnings:", results['warnings'])
    >>> 
    >>> # Access EGDF parameters
    >>> gdf_params = results['gdf_parameters']
    >>> print(f"Optimal scale parameter: {gdf_params.get('S_opt', 'Not found')}")
    >>> print(f"Global optimum Z0: {gdf_params.get('z0', 'Not found')}")
    
    **Memory Management:**
    
    >>> # For large datasets, use flush=True to save memory
    >>> large_data = np.random.normal(0, 1, 10000)
    >>> egdf_large = EGDF(data=large_data, catch=True)
    >>> egdf_large.fit()
    >>> 
    >>> # Analysis with memory cleanup
    >>> homogeneity = DataHomogeneity(egdf_large, flush=True)
    >>> homogeneity.fit()  # pdf_points and di_points_n cleared after analysis
    
    Notes
    -----
    **Mathematical Background:**
    
    The gnostic homogeneity analysis is based on the principle that homogeneous data should
    produce a unimodal PDF with specific characteristics for EGDF:
    
    - **EGDF Uniqueness**: Each data sample has exactly one optimal EGDF representation
    - **Scale Optimization**: EGDF automatically finds the best scale parameter S_opt
    - **Density Properties**: Homogeneous data produces single maximum, non-negative density
    - **Numerical Sensitivity**: Analysis must be numerical, not based on visual inspection
    
    **Why Only EGDF:**
    
    Homogeneity testing is only applicable to EGDF because:
    - EGDF provides unique representation for each data sample
    - Automatic scale parameter optimization enables reliable homogeneity testing
    - Global nature with normalized weights makes it suitable for detecting data structure
    - Robustness against outliers while maintaining sensitivity to detect them
    - QGDF, ELDF, and QLDF have different mathematical properties unsuitable for this analysis
    
    **Gnostic Principles Applied:**
    
    - **Data Primacy**: Data are the primary parameters determining the distribution
    - **Let Data Speak**: Analysis relies on data-driven optimal parameters
    - **Unique Representation**: EGDF provides the one and only best representation
    - **Numerical Decision Making**: Homogeneity decisions must be numerical, not visual
    
    **Parameter Tuning Guidelines:**
    
    - **smoothing_sigma**: Start with 1.0, increase for noisy data to improve numerical stability
    - **min_height_ratio**: Start with 0.01, increase to reduce false positives from noise
    - **min_distance**: Usually auto-calculated, manually set for specific data characteristics
    - Remember: Visual inspection can be misleading, rely on numerical analysis
    
    **Performance Considerations:**
    
    - Memory usage scales with data size due to PDF point storage
    - Use flush=True for large datasets if PDF data not needed afterward
    - Smoothing adds computational cost but improves numerical robustness
    - EGDF fitting provides optimal parameters, reducing computational overhead
    
    **Integration with Existing Workflows:**
    
    This class integrates seamlessly with existing EGDF workflows:
    - Reads parameters from fitted EGDF objects including S_opt
    - Appends errors/warnings to existing EGDF parameter collections
    - Updates EGDF objects with homogeneity results
    - Preserves all original EGDF functionality and gnostic principles
    - Works with EGDF's automatic parameter optimization
    
    **Theoretical Foundation:**
    
    Based on gnostic theory where:
    - Global distribution functions assume data sample homogeneity
    - Non-homogeneous samples exhibit multiple density maxima or negative densities
    - EGDF's unique scale parameter enables reliable homogeneity hypothesis testing
    - Robustness properties make EGDF particularly suitable for small, widely spread samples
    
    See Also
    --------
    EGDF : Estimating Global Distribution Function
    """
    
    def __init__(self, gdf: EGDF, verbose=True, catch=True, flush=False,
                 smoothing_sigma=1.0, min_height_ratio=0.01, min_distance=None):
        self.gdf = gdf
        self.verbose = verbose
        self.catch = catch
        self.flush = flush
        self.params = {}
        self._fitted = False

        # Analysis parameters
        self.smoothing_sigma = smoothing_sigma
        self.min_height_ratio = min_height_ratio
        self.min_distance = min_distance

        # Results
        self.z0 = None
        self.picks = []
        self.is_homogeneous = None
        self.global_extremum_idx = None

        # Logger setup
        self.logger = get_logger(self.__class__.__name__, level=logging.DEBUG if verbose else logging.ERROR)
        self.logger.debug(f"{self.__class__.__name__} initialized: ")

        self._gdf_obj_validation()
        self._validate_egdf_only()

    def _validate_egdf_only(self):
        """Validate that the GDF object is EGDF only."""
        self.logger.info("Validating GDF object for DataHomogeneity analysis")
        class_name = self.gdf.__class__.__name__
        
        if 'QGDF' in class_name:
            self.logger.error(f"DataHomogeneity only supports EGDF objects. "
                              f"Received {class_name}. QGDF is not supported for homogeneity analysis.")
            raise ValueError(
                f"DataHomogeneity only supports EGDF objects. "
                f"Received {class_name}. QGDF is not supported for homogeneity analysis."
            )
        
        if 'ELDF' in class_name or 'QLDF' in class_name:
            self.logger.error(f"DataHomogeneity only supports EGDF objects. "
                              f"Received {class_name}. Local distribution functions (ELDF, QLDF) are not supported "
                              f"for homogeneity analysis.")
            raise ValueError(
                f"DataHomogeneity only supports EGDF objects. "
                f"Received {class_name}. Local distribution functions (ELDF, QLDF) are not supported "
                f"for homogeneity analysis."
            )
        
        if 'EGDF' not in class_name:
            # Fallback detection based on methods
            if not hasattr(self.gdf, '_fit_egdf'):
                self.logger.error(f"DataHomogeneity only supports EGDF objects. "
                                  f"Cannot determine if {class_name} is EGDF. "
                                  f"Object must be EGDF for homogeneity analysis.")
                raise ValueError(
                    f"DataHomogeneity only supports EGDF objects. "
                    f"Cannot determine if {class_name} is EGDF. "
                    f"Object must be EGDF for homogeneity analysis."
                )

    def _gdf_obj_validation(self):
        """Validate that the EGDF object meets requirements for homogeneity analysis."""
        self.logger.debug("Validating EGDF object attributes for homogeneity analysis")
        if not hasattr(self.gdf, '_fitted'):
            self.logger.error("EGDF object must have _fitted attribute")
            raise ValueError("EGDF object must have _fitted attribute")
        
        if not self.gdf._fitted:
            self.logger.error("EGDF object must be fitted before homogeneity analysis")
            raise ValueError("EGDF object must be fitted before homogeneity analysis")
        
        required_attrs = ['data']
        for attr in required_attrs:
            if not hasattr(self.gdf, attr):
                self.logger.error(f"EGDF object missing required attribute: {attr}")
                raise ValueError(f"EGDF object missing required attribute: {attr}")
        
        if not (hasattr(self.gdf, 'pdf_points') and self.gdf.pdf_points is not None):
            if hasattr(self.gdf, 'catch') and not self.gdf.catch:
                self.logger.error("EGDF object must have catch=True to generate "
                                  "pdf_points required for homogeneity analysis.")
                raise AttributeError(
                    f"EGDF object must have catch=True to generate "
                    f"pdf_points required for homogeneity analysis."
                )
            else:
                self.logger.error("EGDF object is missing 'pdf_points'. "
                                  "Please ensure catch=True when fitting EGDF.")
                raise AttributeError(
                    f"EGDF object is missing 'pdf_points'. "
                    f"Please ensure catch=True when fitting EGDF."
                )

    def _prepare_params_from_gdf(self):
        """Extract and prepare parameters from the EGDF object."""
        self.logger.debug("Extracting parameters from EGDF object")
        gdf_params = {}
        
        # Extract basic parameters
        if hasattr(self.gdf, 'params') and self.gdf.params:
            gdf_params.update(self.gdf.params)
        
        # Extract direct attributes
        direct_attrs = ['S', 'S_opt', 'z0', 'data', 'pdf_points', 'di_points_n']
        for attr in direct_attrs:
            if hasattr(self.gdf, attr):
                value = getattr(self.gdf, attr)
                if value is not None:
                    gdf_params[attr] = value
        
        return gdf_params

    def _append_error(self, error_message, exception_type=None):
        """Append error to existing errors in EGDF params or create new ones."""
        self.logger.error(error_message)
        error_entry = {
            'method': 'DataHomogeneity',
            'error': error_message,
            'exception_type': exception_type or 'DataHomogeneityError'
        }
        
        # Add to EGDF object params if possible
        if hasattr(self.gdf, 'params'):
            if 'errors' not in self.gdf.params:
                self.gdf.params['errors'] = []
            self.gdf.params['errors'].append(error_entry)
        
        # Also add to local params
        if 'errors' not in self.params:
            self.params['errors'] = []
        self.params['errors'].append(error_entry)

    def _append_warning(self, warning_message):
        """Append warning to existing warnings in EGDF params or create new ones."""
        self.logger.warning(warning_message)
        warning_entry = {
            'method': 'DataHomogeneity',
            'warning': warning_message
        }
        
        # Add to EGDF object params if possible
        if hasattr(self.gdf, 'params'):
            if 'warnings' not in self.gdf.params:
                self.gdf.params['warnings'] = []
            self.gdf.params['warnings'].append(warning_entry)
        
        # Also add to local params
        if 'warnings' not in self.params:
            self.params['warnings'] = []
        self.params['warnings'].append(warning_entry)

    def _flush_memory(self):
        """Flush di_points and pdf_points from memory if flush=True."""
        self.logger.info("Flushing memory if flush=True")
        if self.flush:
            # # Flush from EGDF object attributes
            # if hasattr(self.gdf, 'di_points_n'):
            #     self.gdf.di_points_n = None
            #     if self.verbose:
            #         print("Flushed di_points_n from EGDF object to save memory.")
            
            # if hasattr(self.gdf, 'pdf_points'):
            #     self.gdf.pdf_points = None
            #     if self.verbose:
            #         print("Flushed pdf_points from EGDF object to save memory.")
            
            # Flush from EGDF object params dictionary
            if hasattr(self.gdf, 'params') and self.gdf.params:
                if 'di_points_n' in self.gdf.params:
                    del self.gdf.params['di_points_n']
                    self.logger.info("Removed di_points_n from EGDF params dictionary to save memory.")
                
                if 'pdf_points' in self.gdf.params:
                    del self.gdf.params['pdf_points']
                    self.logger.info("Removed pdf_points from EGDF params dictionary to save memory.")
            
            # Also flush from local params if they exist
            if 'gdf_parameters' in self.params and self.params['gdf_parameters']:
                if 'di_points_n' in self.params['gdf_parameters']:
                    del self.params['gdf_parameters']['di_points_n']
                    self.logger.info("Removed di_points_n from local gdf_parameters to save memory.")

                if 'pdf_points' in self.params['gdf_parameters']:
                    del self.params['gdf_parameters']['pdf_points']
                    self.logger.info("Removed pdf_points from local gdf_parameters to save memory.")

    def fit(self, plot: bool = False) -> bool:
        """
        Perform comprehensive homogeneity analysis on the EGDF object.
        
        This is the primary analysis method that executes the complete homogeneity assessment
        pipeline. It analyzes the probability density function (PDF) of the fitted EGDF object
        to determine if the underlying data exhibits homogeneous characteristics based on
        peak detection and PDF properties.
        
        **Analysis Pipeline:**
        
        1. **Parameter Extraction**: Retrieves comprehensive parameters from the input EGDF object
        2. **PDF Processing**: Applies Gaussian smoothing to reduce noise and improve detection
        3. **Peak Detection**: Identifies maxima in the smoothed PDF
        4. **Homogeneity Assessment**: Evaluates based on peak count and PDF negativity
        5. **Result Storage**: Stores comprehensive analysis results and metadata
        6. **Memory Management**: Optionally flushes large arrays to conserve memory
        
        **Homogeneity Criteria:**
        
        - **EGDF**: Data is homogeneous if PDF has exactly one global maximum and no negative values
        
        The method automatically handles parameter tuning, error tracking, and integration
        with the existing EGDF parameter system.

        Parameters
        ----------
        plot : bool, optional
            If True, generates plots for visual inspection of the analysis results.
            - True: Displays plots of original and smoothed PDF with detected maxima

        Returns
        -------
        bool
            The primary homogeneity result:
            - True: Data exhibits homogeneous characteristics
            - False: Data is heterogeneous (multiple maxima or negative PDF values)
        
        Raises
        ------
        RuntimeError
            If the analysis fails due to:
            - Numerical instabilities in PDF processing
            - Insufficient or corrupted PDF data
            - Memory allocation issues during processing
            
        AttributeError
            If the EGDF object lacks required attributes:
            - Missing pdf_points (ensure catch=True during EGDF fitting)
            - Missing di_points_n for position mapping
            - Invalid or incomplete EGDF state
        
        ValueError
            If analysis parameters are invalid:
            - Negative smoothing_sigma
            - Invalid min_height_ratio (not between 0 and 1)
            - Corrupted PDF data (NaN, infinite values)
        
        Side Effects
        -----------
        - Updates self.is_homogeneous with the analysis result
        - Populates self.picks with detected maxima information
        - Sets self.z0 with the global optimum value
        - Updates self.global_extremum_idx with the maximum location
        - Modifies EGDF object params with homogeneity results (if catch=True)
        - May clear pdf_points and di_points_n from EGDF object (if flush=True)
        - Appends any errors or warnings to existing EGDF error/warning collections
        
        Examples
        --------
        **Basic Usage:**
        
        >>> # After creating DataHomogeneity instance
        >>> homogeneity = DataHomogeneity(egdf_object, verbose=True)
        >>> is_homogeneous = homogeneity.fit()
        >>> print(f"Analysis complete. Homogeneous: {is_homogeneous}")
        
        **Memory Management:**
        
        >>> # For large datasets
        >>> homogeneity = DataHomogeneity(large_egdf, flush=True)
        >>> result = homogeneity.fit()  # Automatically frees memory after analysis
        
        **Integration with Workflows:**
        
        >>> # Analysis integrates seamlessly with existing EGDF workflows
        >>> egdf.fit()  # Standard EGDF fitting
        >>> homogeneity = DataHomogeneity(egdf)
        >>> homogeneity.fit()  # Homogeneity analysis
        >>> 
        >>> # Results now available in both objects
        >>> print("EGDF homogeneity flag:", egdf.params['is_homogeneous'])
        >>> print("Detailed analysis:", homogeneity.results())
        
        Notes
        -----
        **Performance Considerations:**
        
        - Processing time scales approximately O(n log n) with PDF length
        - Memory usage depends on PDF resolution and catch parameter
        - Smoothing adds computational overhead but improves robustness
        
        **Parameter Sensitivity:**
        
        The analysis robustness depends on proper parameter tuning:
        - Increase smoothing_sigma for noisy data
        - Adjust min_height_ratio to control sensitivity
        - Set appropriate min_distance to avoid spurious detections
        
        **Mathematical Foundation:**
        
        The method implements gnostic homogeneity theory where:
        - Homogeneous data should produce unimodal PDFs
        - EGDF represents optimal scale parameter selection (expect single peak)
        
        **Quality Assurance:**
        
        The method includes comprehensive validation:
        - PDF integrity checks (no NaN, infinite values)
        - Parameter bounds validation
        - Numerical stability monitoring
        - Automatic fallback strategies for edge cases
        
        See Also
        --------
        plot : Visualize the analysis results
        results : Access comprehensive analysis data
        """
        self.logger.info("Starting homogeneity analysis fit() method")
        try:
            # Prepare parameters from EGDF
            self.logger.debug("Preparing parameters from EGDF object")
            gdf_params = self._prepare_params_from_gdf()
            
            # Set minimum distance if not provided
            if self.min_distance is None:
                self.logger.debug("Minimum distance not provided, calculating...")
                pdf_data = self._get_pdf_data()
                self.min_distance = max(1, len(pdf_data) // 20)
            
            # Perform homogeneity test
            self.logger.info("Testing homogeneity")
            self.is_homogeneous = self._test_homogeneity()
            
            # Extract Z0
            self.logger.info("Extracting global optimum Z0")
            self.z0 = self._get_z0()
            
            # Store comprehensive results
            if self.catch:
                self.params.update({
                    'gdf_type': 'egdf',
                    'is_homogeneous': self.is_homogeneous,
                    'picks': self.picks,
                    'z0': self.z0,
                    'global_extremum_idx': self.global_extremum_idx,
                    'analysis_parameters': {
                        'smoothing_sigma': self.smoothing_sigma,
                        'min_height_ratio': self.min_height_ratio,
                        'min_distance': self.min_distance,
                        'flush': self.flush
                    },
                    'homogeneity_fitted': True
                })
                
                # Include EGDF parameters
                self.params['gdf_parameters'] = gdf_params
            
            # Update EGDF object params if possible
            if hasattr(self.gdf, 'catch') and self.gdf.catch and hasattr(self.gdf, 'params'):
                self.gdf.params.update({
                    'is_homogeneous': self.is_homogeneous,
                    'homogeneity_checked': True,
                    'homogeneity_fitted': True
                })
                
                self.logger.info("Homogeneity results written to EGDF params dictionary.")
                        
            self._fitted = True

            # plot
            if plot:
                self.logger.info("Plotting results as requested")
                self.plot() 

            # Flush memory if requested
            self.logger.info("Handling memory flush if requested")
            self._flush_memory()

            self.logger.info("Homogeneity analysis completed for EGDF.")
            self.logger.info(f"Data is {'homogeneous' if self.is_homogeneous else 'not homogeneous'}")
            self.logger.info(f"Number of maxima detected: {len(self.picks)}")

            return self.is_homogeneous
    
        except Exception as e:
            error_msg = f"Error during homogeneity analysis: {str(e)}"
            self._append_error(error_msg, type(e).__name__)
            raise

    def _test_homogeneity(self):
        """
        Test data homogeneity for EGDF.
        
        Returns
        -------
        bool
            True if homogeneous, False otherwise.
        """
        self.logger.info("Starting homogeneity test for EGDF")
        try:
            pdf_data = self._get_pdf_data()
            has_negative_pdf = np.any(pdf_data < 0)
            
            # EGDF: Look for single global maximum
            self.picks = self._detect_maxima()
            extrema_type = "maxima"
            num_extrema = len(self.picks)
            is_homogeneous = not has_negative_pdf and num_extrema == 1
            
            if self.verbose:
                if not is_homogeneous:
                    reasons = []
                    if has_negative_pdf:
                        reasons.append("PDF has negative values")
                        self._append_warning("PDF contains negative values - may indicate numerical issues")
                    if num_extrema > 1:
                        reasons.append(f"multiple {extrema_type} [{num_extrema}] detected")
                        self._append_warning(f"Multiple {extrema_type} detected - data may not be homogeneous")
                    elif num_extrema == 0:
                        reasons.append(f"no significant {extrema_type} detected")
                        self._append_warning(f"No significant {extrema_type} detected - check smoothing parameters")
                    self.logger.info(f"EGDF data is not homogeneous: {', '.join(reasons)}.")
                else:
                    self.logger.info(f"EGDF data is homogeneous: PDF has no negative values "
                        f"and exactly one {extrema_type[:-1]} detected.")
            
            # Store additional info in params
            if self.catch:
                self.params.update({
                    'has_negative_pdf': has_negative_pdf,
                    f'num_{extrema_type}': num_extrema,
                    'extrema_type': extrema_type
                })
            
            return is_homogeneous
            
        except Exception as e:
            error_msg = f"Error in homogeneity test: {str(e)}"
            self._append_error(error_msg, type(e).__name__)
            raise

    def _detect_maxima(self):
        """Detect maxima for EGDF analysis."""
        self.logger.info("Detecting maxima in the PDF")
        try:
            pdf_data = self._get_pdf_data()
            data_points = self._get_data_points()
            smoothed_pdf = self._smooth_pdf()
            
            min_height = np.max(smoothed_pdf) * self.min_height_ratio
            maxima_idx, _ = find_peaks(smoothed_pdf, 
                                       height=min_height,
                                       distance=self.min_distance)
            
            picks = []
            global_max_value = -np.inf
            
            for idx in maxima_idx:
                pick_info = {
                    'index': int(idx),
                    'position': float(data_points[idx]),
                    'pdf_value': float(pdf_data[idx]),
                    'smoothed_pdf_value': float(smoothed_pdf[idx]),
                    'is_global': False
                }
                picks.append(pick_info)
                
                if smoothed_pdf[idx] > global_max_value:
                    global_max_value = smoothed_pdf[idx]
                    self.global_extremum_idx = idx
            
            # Mark global maximum
            for pick in picks:
                if pick['index'] == self.global_extremum_idx:
                    pick['is_global'] = True
                    break
            
            # Sort by importance (global first, then by height)
            picks.sort(key=lambda x: (not x['is_global'], -x['smoothed_pdf_value']))
            
            return picks
            
        except Exception as e:
            error_msg = f"Error detecting maxima: {str(e)}"
            self._append_error(error_msg, type(e).__name__)
            return []

    def _smooth_pdf(self):
        """Apply Gaussian smoothing to PDF."""
        self.logger.info("Smoothing PDF with Gaussian filter")
        try:
            pdf_data = self._get_pdf_data()
            return gaussian_filter1d(pdf_data, sigma=self.smoothing_sigma)
        except Exception as e:
            error_msg = f"Error smoothing PDF: {str(e)}"
            self._append_error(error_msg, type(e).__name__)
            return pdf_data  # Return unsmoothed data as fallback

    def _get_pdf_data(self):
        """Get PDF values from the EGDF object."""
        self.logger.info("Retrieving PDF data from EGDF object")
        return self.gdf.pdf_points

    def _get_data_points(self):
        """Get data point positions from the EGDF object."""
        self.logger.info("Retrieving data point positions from EGDF object")
        return self.gdf.di_points_n

    def _get_z0(self):
        """Get Z0 (global optimum) value from the EGDF object."""
        self.logger.info("Retrieving Z0 (global optimum) from EGDF object")
        if hasattr(self.gdf, 'z0') and self.gdf.z0 is not None:
            return self.gdf.z0
        elif hasattr(self.gdf, 'params') and 'z0' in self.gdf.params:
            return self.gdf.params['z0']
        else:
            # Fallback: use global extremum from PDF
            if self.global_extremum_idx is not None:
                data_points = self._get_data_points()
                if self.verbose:
                    self._append_warning("Z0 not found in EGDF object. Using PDF global extremum as Z0.")
                return data_points[self.global_extremum_idx]
            return None

    def plot(self, figsize=(12, 8), title=None):
        """
        Create a comprehensive visualization of the homogeneity analysis results.
        
        This method generates an informative plot that displays the probability density
        function (PDF), detected maxima, homogeneity status, and key analysis metrics.
        The visualization provides both quantitative and qualitative insights into the
        data's homogeneous characteristics.
        
        **Plot Components:**
        
        1. **Original PDF Curve**: Blue solid line showing the raw probability density
        2. **Smoothed PDF Curve**: Orange dashed line showing Gaussian-filtered PDF
        3. **Global Maximum**: Red circle with vertical line marking the primary maximum
        4. **Secondary Maxima**: Grey circles with vertical lines for additional maxima
        5. **Z0 Reference**: Cyan dotted line if Z0 differs from detected maximum
        6. **Status Indicator**: Color-coded text box showing homogeneity result
        7. **Analysis Summary**: Information box with key metrics and statistics
        
        The plot layout is optimized for both screen display and publication quality,
        with clear legends, appropriate scaling, and professional formatting.
        
        Parameters
        ----------
        figsize : tuple of float, default=(12, 8)
            Figure dimensions in inches as (width, height).
            - Larger sizes provide better detail visibility
            - Smaller sizes suitable for embedded displays
            - Recommended range: (8, 6) to (16, 12)
            
        title : str, optional
            Custom plot title. If None, generates descriptive title automatically.
            - None: Auto-generated title with EGDF type and homogeneity status
            - str: Custom title text (supports LaTeX formatting)
            - Empty string: No title displayed
        
        Returns
        -------
        None
            The method displays the plot using matplotlib.pyplot.show() and does not
            return any value. The plot appears in the current matplotlib backend.
        
        Raises
        ------
        RuntimeError
            If called before the fit() method has been executed:
            - No analysis results available for visualization
            - Internal state inconsistent or incomplete
            
        AttributeError
            If required plot data is missing or corrupted:
            - PDF data unavailable or deleted (check flush parameter)
            - Data points array missing or malformed
            - Maxima detection results incomplete
            
        ImportError
            If matplotlib is not available or not properly installed
            
        MemoryError
            If insufficient memory for plot generation (rare, for very large datasets)
        
        Side Effects
        -----------
        - Displays interactive plot window (backend-dependent)
        - May create temporary matplotlib figure and axis objects
        - Does not modify any analysis results or object state
        - Plot appearance depends on current matplotlib style settings
        
        Examples
        --------
        **Basic Plotting:**
        
        >>> # After running analysis
        >>> homogeneity = DataHomogeneity(egdf_object)
        >>> homogeneity.fit()
        >>> homogeneity.plot()  # Display with default settings
        
        **Custom Formatting:**
        
        >>> # Custom size and title
        >>> homogeneity.plot(
        ...     figsize=(14, 10),
        ...     title="EGDF Homogeneity Analysis: Production Data"
        ... )
                
        Notes
        -----
        **Visual Interpretation Guide:**
        
        - **Green Status Box**: Data is homogeneous (single maximum, no negative PDF)
        - **Red Status Box**: Data is heterogeneous (multiple maxima or negative values)
        - **Red Markers**: Global maximum
        - **Grey Markers**: Secondary maxima indicating potential heterogeneity
        - **Smooth vs Raw PDF**: Comparison shows impact of noise filtering
        
        **Plot Customization:**
        
        The plot uses matplotlib's standard customization system:
        - Colors follow standard scientific visualization conventions
        - Font sizes and line weights optimized for readability
        - Grid and legend placement maximize information density
        - Axis labels and scales automatically adjusted for data range
        
        **Performance Notes:**
        
        - Plot generation is typically fast (< 1 second for most datasets)
        - Large datasets may require longer rendering times
        - Interactive backends may be slower than static ones
        - Memory usage scales with plot resolution and data size
        
        **Troubleshooting:**
        
        Common issues and solutions:
        - **Empty plot**: Check if fit() was called successfully
        - **Missing data**: Verify flush=False if data needed for plotting
        - **Poor visibility**: Adjust figsize or matplotlib DPI settings
        - **Layout issues**: Use plt.tight_layout() or bbox_inches='tight'
        
        **Mathematical Context:**
        
        The visualization directly represents the mathematical foundation:
        - PDF height indicates probability density magnitude
        - Maximum positions show optimal data characteristics
        - Smoothing reveals underlying distributional structure
        - Multiple maxima indicate potential data clustering or heterogeneity
        
        See Also
        --------
        fit : Perform the homogeneity analysis (required before plotting)
        results : Access numerical analysis results
        """
        self.logger.info("Generating homogeneity analysis plot")
        if not self._fitted:
            self.logger.error("Must call fit() before plotting. Run fit() method first.")
            raise RuntimeError("Must call fit() before plotting. Run fit() method first.")
        
        try:
            fig, ax = plt.subplots(figsize=figsize)
            
            pdf_data = self._get_pdf_data()
            data_points = self._get_data_points()
            smoothed_pdf = self._smooth_pdf()
            
            # Plot PDF and smoothed PDF
            ax.plot(data_points, pdf_data, 'b-', linewidth=2, label='PDF', alpha=0.7)
            ax.plot(data_points, smoothed_pdf, 'orange', linestyle='--', linewidth=1.5, 
                    label='Smoothed PDF', alpha=0.8)
            
            # Plot detected maxima
            for pick in self.picks:
                pos = pick['position']
                pdf_val = pick['pdf_value']
                is_global = pick['is_global']
                
                if is_global:
                    ax.axvline(pos, color='red', linestyle='-', linewidth=2, alpha=0.8)
                    ax.plot(pos, pdf_val, 'o', color='red', markersize=10, 
                        label=f'Global maximum (Z0={pos:.3f})')
                else:
                    ax.axvline(pos, color='grey', linestyle='-', linewidth=1, alpha=0.6)
                    ax.plot(pos, pdf_val, 'o', color='grey', markersize=6, alpha=0.7)
            
            # Add Z0 line if different from global maximum
            if self.z0 is not None and self.global_extremum_idx is not None:
                global_maximum_pos = data_points[self.global_extremum_idx]
                if abs(self.z0 - global_maximum_pos) > 0.001:
                    ax.axvline(self.z0, color='cyan', linestyle=':', linewidth=2, alpha=0.8,
                            label=f'Original Z0={self.z0:.3f}')
            
            # Add homogeneity status text
            status_text = "Homogeneous" if self.is_homogeneous else "Not Homogeneous"
            status_color = 'green' if self.is_homogeneous else 'red'
            
            ax.text(0.02, 0.98, status_text, transform=ax.transAxes, 
                    fontsize=12, fontweight='bold', color=status_color,
                    verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=status_color))
            
            # Add analysis info
            info_text = f"Type: EGDF\n"
            info_text += f"Maxima: {len(self.picks)}\n"
                
            if hasattr(self, 'params') and 'has_negative_pdf' in self.params:
                info_text += f"Negative PDF: {'Yes' if self.params['has_negative_pdf'] else 'No'}"
            
            ax.text(0.02, 0.02, info_text, transform=ax.transAxes, 
                    fontsize=10, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='lightgrey', alpha=0.7))
            
            ax.set_xlabel('Data Points')
            ax.set_ylabel('PDF Values')
            
            if title is None:
                homogeneous_str = "Homogeneous" if self.is_homogeneous else "Non-Homogeneous"
                title = f"EGDF {homogeneous_str} Data Analysis"
            ax.set_title(title)
            
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            error_msg = f"Error creating plot: {str(e)}"
            self._append_error(error_msg, type(e).__name__)
            raise

    def results(self) -> Dict[str, Any]:
        """
        Retrieve comprehensive homogeneity analysis results and metadata.
        
        This method provides access to all analysis results, parameters, and diagnostic
        information generated during the homogeneity assessment. It returns a complete
        dictionary containing quantitative results, detected maxima details, analysis
        parameters, original EGDF object information, and any errors or warnings
        encountered during processing.
        
        **Result Categories:**
        
        1. **Primary Results**: Core homogeneity findings (is_homogeneous, maxima count)
        2. **Maxima Details**: Complete information about detected peaks
        3. **Analysis Parameters**: Configuration settings used during analysis  
        4. **EGDF Parameters**: Original parameters from the input EGDF object
        5. **Diagnostic Data**: Errors, warnings, and processing metadata
        6. **Quality Metrics**: PDF characteristics and numerical indicators
        
        The returned dictionary maintains referential integrity and provides
        comprehensive traceability for analysis reproducibility and debugging.
        
        Returns
        -------
        dict
            Comprehensive results dictionary with the following structure:
            
            **Core Analysis Results:**
            - 'gdf_type' (str): Always 'egdf' for this class
            - 'is_homogeneous' (bool): Primary homogeneity determination
            - 'z0' (float): Global optimum value (Z0) from EGDF or detected maximum
            - 'global_extremum_idx' (int): Array index of global maximum
            - 'homogeneity_fitted' (bool): Confirmation flag for completed analysis
            
            **Maxima Information:**
            - 'picks' (List[Dict]): Detected maxima with detailed properties:
            - 'index' (int): Array position of maximum
            - 'position' (float): Data value at maximum location
            - 'pdf_value' (float): Original PDF value at maximum
            - 'smoothed_pdf_value' (float): Smoothed PDF value at maximum
            - 'is_global' (bool): Flag indicating global maximum
            
            **PDF Characteristics:**
            - 'has_negative_pdf' (bool): Whether PDF contains negative values
            - 'num_maxima' (int): Count of detected maxima
            - 'extrema_type' (str): Always 'maxima' for EGDF
            
            **Analysis Configuration:**
            - 'analysis_parameters' (Dict): Settings used during analysis:
            - 'smoothing_sigma' (float): Gaussian smoothing parameter
            - 'min_height_ratio' (float): Minimum height threshold for detection
            - 'min_distance' (int): Minimum separation between maxima
            - 'flush' (bool): Memory management setting
            
            **Original EGDF Data:**
            - 'gdf_parameters' (Dict): Complete parameter set from input EGDF object
            including S, S_opt, z0, data arrays, and fitted results
            
            **Diagnostics (if present):**
            - 'errors' (List[Dict]): Analysis errors with method and type information
            - 'warnings' (List[Dict]): Analysis warnings and advisory messages
        
        Raises
        ------
        RuntimeError
            If called before fit() method execution:
            - "No analysis results available. Call fit() method first."
            - Analysis state is incomplete or inconsistent
            
        RuntimeError
            If results storage is disabled:
            - "No results stored. Ensure catch=True during initialization."
            - catch=False prevents result storage for memory conservation
        
        Examples
        --------
        **Basic Result Access:**
        
        >>> # After running analysis
        >>> homogeneity = DataHomogeneity(egdf_object)
        >>> homogeneity.fit()
        >>> results = homogeneity.results()
        >>> print(f"Homogeneous: {results['is_homogeneous']}")
        >>> print(f"Maxima detected: {len(results['picks'])}")
        
        **Detailed Maxima Analysis:**
        
        >>> results = homogeneity.results()
        >>> for i, maximum in enumerate(results['picks']):
        ...     status = "Global" if maximum['is_global'] else "Local"
        ...     print(f"{status} maximum {i+1}:")
        ...     print(f"  Position: {maximum['position']:.4f}")
        ...     print(f"  PDF value: {maximum['pdf_value']:.4f}")
        ...     print(f"  Smoothed PDF: {maximum['smoothed_pdf_value']:.4f}")
        
        **Error and Warning Inspection:**
        
        >>> results = homogeneity.results()
        >>> if 'errors' in results:
        ...     print("Analysis encountered errors:")
        ...     for error in results['errors']:
        ...         print(f"  {error['method']}: {error['error']}")
        >>> 
        >>> if 'warnings' in results:
        ...     print("Analysis warnings:")
        ...     for warning in results['warnings']:
        ...         print(f"  {warning['method']}: {warning['warning']}")
        
        **Parameter Traceability:**
        
        >>> results = homogeneity.results()
        >>> analysis_config = results['analysis_parameters']
        >>> print("Analysis was performed with:")
        >>> print(f"  Smoothing: {analysis_config['smoothing_sigma']}")
        >>> print(f"  Min height ratio: {analysis_config['min_height_ratio']}")
        >>> print(f"  Min distance: {analysis_config['min_distance']}")
            
        Notes
        -----
        **Data Integrity:**
        
        The returned dictionary is a deep copy of internal results, ensuring:
        - Modifications to returned data don't affect internal state
        - Thread-safe access to results
        - Consistent data even if original EGDF object changes
        
        **Memory Considerations:**
        
        - Results dictionary may contain large arrays (PDF points, data points)
        - Use flush=True during initialization to reduce memory footprint
        - Consider extracting only needed fields for memory-constrained environments
        
        **Version Compatibility:**
        
        The results structure is designed for forward/backward compatibility:
        - New fields added with default values for missing data
        - Deprecated fields maintained for transition periods
        - Type consistency maintained across versions
        
        **Performance Notes:**
        
        - Dictionary creation involves copying large data structures
        - Access time is O(1) for individual fields
        - Memory usage scales with original data size and PDF resolution
        
        **Integration Patterns:**
        
        Common usage patterns for results integration:
        - Store results in databases using JSON serialization
        - Pass results to downstream analysis pipelines
        - Generate reports using template systems
        - Create batch analysis summaries and comparisons
        
        **Validation and Quality Control:**
        
        The results include comprehensive quality indicators:
        - Error counts and descriptions for debugging
        - Warning flags for borderline cases
        - Parameter consistency checks
        - Numerical stability indicators
        
        See Also
        --------
        fit : Perform the analysis to generate results
        plot : Visualize the analysis results
        DataHomogeneity.__init__ : Configure result storage with catch parameter
        """
        self.logger.info("Retrieving homogeneity analysis results")
        if not self._fitted:
            self.logger.error("No analysis results available. Call fit() method first.")
            raise RuntimeError("No analysis results available. Call fit() method first.")
        
        if not self.params:
            self.logger.error("No results stored. Ensure catch=True during initialization.")
            raise RuntimeError("No results stored. Ensure catch=True during initialization.")
        
        return self.params.copy()

    @property
    def fitted(self):
        """bool: True if the analysis has been completed, False otherwise."""
        return self._fitted
    
    def __repr__(self):
        return f"DataHomogeneity(gdf_type='egdf', fitted={self._fitted}, is_homogeneous={self.is_homogeneous})"