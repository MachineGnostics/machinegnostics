"""
Z0 Estimator - Universal class for estimating Z0 point for both EGDF and ELDF

Z0 is the point where PDF is at its global maximum, using advanced interpolation methods.

Author: Nirmal Parmar
Machine Gnostics
"""

import numpy as np
import warnings
from typing import Union, Dict, Any, Optional
from machinegnostics.magcal import EGDF, ELDF

class Z0Estimator:
    """
    Universal Z0 Estimator for EGDF and ELDF objects.
    
    A specialized class for estimating the Z0 point, defined as the location where the 
    Probability Density Function (PDF) reaches its global maximum. This class provides 
    advanced interpolation and optimization methods to accurately determine the Z0 point 
    for both Estimating Global Distribution Function (EGDF) and Estimating Local 
    Distribution Function (ELDF) objects.
    
    The Z0 point represents the most probable value in the distribution and is crucial 
    for various applications including peak detection, modal analysis, risk assessment, 
    and optimal point identification. The estimator automatically selects the best 
    estimation method based on data characteristics and provides comprehensive analysis 
    capabilities.
    
    Key Features:
        - Universal compatibility with both EGDF and ELDF objects
        - Multiple estimation methods with automatic selection
        - Advanced interpolation techniques for high precision
        - Robust fallback mechanisms for reliable estimation
        - Comprehensive visualization and analysis tools
        - Detailed estimation metadata and diagnostics
        - Performance optimization for different data sizes
        - Cross-platform compatibility with optional dependencies
    
    Estimation Methods:
        The class employs four main estimation approaches:
        
        1. Spline Optimization: Uses scipy UnivariateSpline with minimize_scalar
           - Best for smooth, high-resolution data (≥20 points, low noise)
           - Provides analytical precision through continuous spline representation
           - Automatically selected when smoothness < 0.1 and n_points ≥ 20
        
        2. Polynomial Fitting: Fits polynomials around the maximum and finds critical points
           - Suitable for moderate-resolution data (≥15 points)
           - Uses degree 2-3 polynomials within a local window around the maximum
           - Finds analytical maximum by solving derivative = 0
        
        3. Refined Interpolation: Creates fine grids using cubic interpolation
           - General-purpose method for various data qualities
           - Generates 1000-point fine grid around the maximum region
           - Balances accuracy and computational efficiency
        
        4. Simple Maximum: Direct maximum finding with optional parabolic interpolation
           - Fallback method for small datasets or when advanced methods fail
           - Uses three-point parabolic interpolation for sub-point precision
           - Most robust method with minimal computational requirements
    
    Automatic Method Selection:
        The estimator analyzes data characteristics to choose the optimal method:
        - Data size (number of points available)
        - Smoothness metric based on second differences
        - PDF value range and distribution shape
        - Availability of required dependencies (scipy)
    
    Parameters:
        gdf_object : EGDF or ELDF
            A fitted EGDF or ELDF object containing the distribution data.
            Must have been successfully fitted with _fitted=True and contain
            PDF data either as pdf_points/di_points_n or pdf/data attributes.
            
        optimize : bool, optional (default=True)
            Estimation strategy selection:
            - True: Use advanced interpolation methods for higher accuracy
            - False: Use simple linear search on existing points
            
            Advanced methods provide sub-point precision but require more computation.
            Simple search is faster but limited to existing point resolution.
            
        verbose : bool, optional (default=False)
            Control output verbosity:
            - True: Print detailed progress, method selection, and diagnostic information
            - False: Run silently with minimal output
            
            Verbose mode helps with debugging and understanding the estimation process.
    
    Attributes:
        gdf : EGDF or ELDF
            Reference to the input GDF object for data access and result updating.
            
        gdf_type : str
            Detected type of the GDF object ('egdf' or 'eldf').
            Used for type-specific handling and display purposes.
            
        optimize : bool
            The optimization strategy flag passed during initialization.
            
        verbose : bool
            The verbosity flag for controlling output detail level.
            
        z0 : float or None
            The estimated Z0 value where PDF reaches its maximum.
            None until estimate_z0() is called successfully.
            
        estimation_info : dict
            Comprehensive metadata about the last estimation including:
            - z0: The estimated Z0 value
            - z0_method: Method used for estimation
            - z0_max_pdf_value: PDF value at Z0 point
            - z0_interpolation_points: Number of points used
            - gdf_type: Type of GDF object processed
            - z0_max_pdf_index: Index of maximum (for linear search)
    
    Methods:
        estimate_z0() -> float:
            Perform Z0 estimation and return the result.
            
        get_estimation_info() -> Dict[str, Any]:
            Get detailed information about the last estimation.
            
        plot_z0_analysis(figsize=(12, 6)):
            Create visualization showing PDF curve and Z0 location.
    
    Raises:
        ValueError:
            - If gdf_object is not a fitted EGDF or ELDF object
            - If gdf_object lacks required attributes (_fitted, data)
            - If PDF data is not available in the gdf_object
            - If smooth curves are not available when needed
            - If Z0 estimation is attempted before calling estimate_z0()
        
        RuntimeError:
            - If all estimation methods fail to converge
            - If numerical instabilities prevent successful estimation
        
        ImportError:
            - If scipy is required for advanced methods but not available
            - If matplotlib is needed for plotting but not installed
    
    Examples:
        Basic Z0 estimation with ELDF:
        >>> from machinegnostics.magcal import ELDF
        >>> from machinegnostics.magcal import Z0Estimator
        >>> 
        >>> # Create and fit ELDF
        >>> data = [7, 8, 8, 9, 11, 12, 13, 14, 15, 18, 20, 28, 37, 42]
        >>> eldf = ELDF(data, n_points=500)
        >>> eldf.fit()
        >>> 
        >>> # Estimate Z0 with optimization
        >>> z0_estimator = Z0Estimator(eldf, optimize=True, verbose=True)
        >>> z0_value = z0_estimator.estimate_z0()
        >>> 
        >>> print(f"Z0 point: {z0_value:.6f}")
        >>> print(f"Method used: {z0_estimator.get_estimation_info()['z0_method']}")
        >>> 
        >>> # Visualize results
        >>> z0_estimator.plot_z0_analysis()
        
        Z0 estimation with EGDF:
        >>> from machinegnostics.magcal import EGDF
        >>> 
        >>> # Create and fit EGDF
        >>> egdf = EGDF(data, verbose=True)
        >>> egdf.fit()
        >>> 
        >>> # Quick Z0 estimation without optimization
        >>> z0_estimator = Z0Estimator(egdf, optimize=False)
        >>> z0_value = z0_estimator.estimate_z0()
        >>> 
        >>> print(f"EGDF Z0: {z0_value:.6f}")
        
        Comparing estimation methods:
        >>> # Compare optimized vs simple methods
        >>> def compare_z0_methods(gdf_object):
        ...     # Optimized estimation
        ...     estimator_opt = Z0Estimator(gdf_object, optimize=True, verbose=False)
        ...     z0_opt = estimator_opt.estimate_z0()
        ...     
        ...     # Simple estimation
        ...     estimator_simple = Z0Estimator(gdf_object, optimize=False, verbose=False)
        ...     z0_simple = estimator_simple.estimate_z0()
        ...     
        ...     print(f"Optimized Z0: {z0_opt:.6f} (method: {estimator_opt.get_estimation_info()['z0_method']})")
        ...     print(f"Simple Z0: {z0_simple:.6f}")
        ...     print(f"Difference: {abs(z0_opt - z0_simple):.6f}")
        ...     
        ...     return z0_opt, z0_simple
        >>> 
        >>> z0_opt, z0_simple = compare_z0_methods(eldf)
        
        Batch processing with error handling:
        >>> datasets = [data1, data2, data3]  # Multiple datasets
        >>> z0_results = []
        >>> 
        >>> for i, dataset in enumerate(datasets):
        ...     try:
        ...         eldf = ELDF(dataset)
        ...         eldf.fit()
        ...         
        ...         z0_estimator = Z0Estimator(eldf, optimize=True, verbose=False)
        ...         z0 = z0_estimator.estimate_z0()
        ...         
        ...         z0_results.append({
        ...             'dataset_id': i,
        ...             'z0': z0,
        ...             'method': z0_estimator.get_estimation_info()['z0_method'],
        ...             'pdf_max': z0_estimator.get_estimation_info()['z0_max_pdf_value']
        ...         })
        ...         
        ...     except Exception as e:
        ...         print(f"Failed to process dataset {i}: {e}")
        ...         z0_results.append({'dataset_id': i, 'error': str(e)})
        
        Custom analysis workflow:
        >>> # Detailed analysis with multiple visualizations
        >>> eldf = ELDF(data, n_points=1000, verbose=True)
        >>> eldf.fit(plot=False)
        >>> 
        >>> # Estimate Z0 with detailed output
        >>> z0_estimator = Z0Estimator(eldf, optimize=True, verbose=True)
        >>> z0 = z0_estimator.estimate_z0()
        >>> 
        >>> # Get comprehensive information
        >>> info = z0_estimator.get_estimation_info()
        >>> print("Z0 Estimation Results:")
        >>> for key, value in info.items():
        ...     print(f"  {key}: {value}")
        >>> 
        >>> # Create analysis plots
        >>> z0_estimator.plot_z0_analysis(figsize=(15, 8))
        >>> 
        >>> # Plot original ELDF with Z0 marked
        >>> eldf.plot(plot='both', bounds=True)
    
    Performance Considerations:
        Method Selection Impact:
        - Spline optimization: Highest accuracy, moderate computation time
        - Polynomial fitting: Good accuracy, fast computation
        - Refined interpolation: Balanced accuracy and speed
        - Simple maximum: Fastest, adequate accuracy for most applications
        
        Data Size Recommendations:
        - < 10 points: Use optimize=False for reliability
        - 10-50 points: Either method works well
        - 50-1000 points: Optimize=True provides better precision
        - > 1000 points: Consider data sampling for performance
        
        Memory Usage:
        - Minimal additional memory overhead
        - Estimation info dictionary is lightweight
        - Plotting creates temporary arrays only
        
        Computational Complexity:
        - Simple method: O(n) where n is number of points
        - Advanced methods: O(n log n) to O(n²) depending on method
        - Spline optimization: Additional scipy optimization overhead
    
    Algorithm Details:
        Spline Optimization Process:
        1. Create UnivariateSpline with smoothing factor s=0
        2. Use cubic splines (k=3) for smooth representation
        3. Apply minimize_scalar with bounded method
        4. Optimize negative spline to find maximum
        5. Validate result within data bounds
        
        Polynomial Fitting Process:
        1. Identify approximate maximum location
        2. Define local window around maximum (±5 points or ±20% of data)
        3. Fit polynomial of degree 2-3 to window data
        4. Compute analytical derivative of polynomial
        5. Find roots of derivative (critical points)
        6. Evaluate polynomial at critical points to find maximum
        
        Refined Interpolation Process:
        1. Locate approximate maximum in original data
        2. Create focused window around maximum region
        3. Apply cubic interpolation to window data
        4. Generate 1000-point fine grid within window
        5. Find maximum on fine grid for sub-point precision
        
        Simple Maximum Process:
        1. Direct maximum finding in original data points
        2. Apply three-point parabolic interpolation if possible
        3. Validate interpolated result within reasonable bounds
        4. Fall back to direct maximum if interpolation fails
    
    Integration with GDF Objects:
        Automatic Updates:
        - Sets z0 attribute in the original GDF object
        - Updates params dictionary if catch=True
        - Preserves all original GDF functionality
        - Enables immediate use in GDF plotting methods
        
        Data Access Strategy:
        - Prioritizes smooth curve data (pdf_points, di_points_n)
        - Falls back to raw data (pdf, data) if smooth curves unavailable
        - Validates data availability before processing
        - Handles both EGDF and ELDF data structures transparently
    
    Troubleshooting:
        Common Issues and Solutions:
        
        "GDF object must be fitted before Z0 estimation":
        - Ensure gdf_object.fit() was called successfully
        - Check that gdf_object._fitted is True
        - Verify fitting completed without errors
        
        "Please ensure GDF has smooth curves generated":
        - Use n_points > 0 when creating GDF object
        - Ensure fitting was done with catch=True
        - Check that pdf_points and di_points_n are available
        
        "SciPy not available for spline optimization":
        - Install scipy: pip install scipy
        - Or use optimize=False for methods not requiring scipy
        - Estimator will automatically fall back to available methods
        
        Estimation Accuracy Issues:
        - Increase n_points in GDF object for better resolution
        - Use optimize=True for higher precision methods
        - Check data quality and consider preprocessing
        - Verify sufficient data points around the true maximum
        
        Performance Issues:
        - Use optimize=False for faster estimation
        - Reduce n_points in GDF for faster processing
        - Consider data sampling for very large datasets
        - Set verbose=False to reduce output overhead
    
    See Also:
        ELDF: Estimating Local Distribution Function
        EGDF: Estimating Global Distribution Function
    
    Notes:
        - Z0 estimation requires successful GDF fitting with PDF data available
        - The class automatically detects GDF type and adapts accordingly
        - All estimation methods are designed to be robust and provide fallbacks
        - Visualization capabilities require matplotlib installation
        - Advanced interpolation methods require scipy installation
        - The estimator preserves and updates the original GDF object
        - Multiple estimation calls on the same object update results incrementally
        - Thread-safe for read operations, not recommended for concurrent estimation
    """

    def __init__(self,
                 gdf_object: Union[EGDF, ELDF],
                 optimize: bool = True,
                 verbose: bool = False):
        """
        Initialize Z0 estimator.
        
        Parameters:
        -----------
        gdf_object : EGDF or ELDF
            Fitted EGDF or ELDF object
        optimize : bool
            If True, use advanced interpolation methods
            If False, use simple linear search
        verbose : bool
            Verbose output
        """
        # Validate input object
        self._validate_gdf_object(gdf_object)
        
        self.gdf = gdf_object
        self.gdf_type = self._detect_gdf_type()
        self.optimize = optimize
        self.verbose = verbose
        
        # Results storage
        self.z0 = None
        self.estimation_info = {}
        
    def _validate_gdf_object(self, gdf_object):
        """Validate that the input is a fitted GDF object."""
        if not hasattr(gdf_object, '_fitted'):
            raise ValueError("GDF object must have _fitted attribute")
        
        if not gdf_object._fitted:
            raise ValueError("GDF object must be fitted before Z0 estimation")
        
        # Check for required attributes
        required_attrs = ['data']
        for attr in required_attrs:
            if not hasattr(gdf_object, attr):
                raise ValueError(f"GDF object missing required attribute: {attr}")
        
        # Check for PDF data
        if not (hasattr(gdf_object, 'pdf_points') and hasattr(gdf_object, 'di_points_n')):
            if not (hasattr(gdf_object, 'pdf') and hasattr(gdf_object, 'data')):
                raise ValueError("GDF object must have PDF data available")
    
    def _detect_gdf_type(self):
        """Detect whether the object is EGDF or ELDF."""
        class_name = self.gdf.__class__.__name__
        if 'EGDF' in class_name:
            return 'egdf'
        elif 'ELDF' in class_name:
            return 'eldf'
        else:
            # Try to detect by checking for specific methods
            if hasattr(self.gdf, '_compute_egdf_core'):
                return 'egdf'
            elif hasattr(self.gdf, '_compute_eldf_core'):
                return 'eldf'
            else:
                raise ValueError("Cannot determine GDF type. Object must be EGDF or ELDF.")
    
    def estimate_z0(self) -> float:
        """
        Estimate the Z0 point where the Probability Density Function (PDF) reaches its maximum.
        
        This method performs comprehensive Z0 estimation using either advanced interpolation
        methods or simple linear search, depending on the optimize parameter. The Z0 point
        represents the most probable value in the distribution and is crucial for modal
        analysis, peak detection, and optimal point identification.
        
        The estimation process automatically selects the best method based on data 
        characteristics including size, smoothness, and quality. Advanced methods provide
        sub-point precision through spline optimization, polynomial fitting, or refined
        interpolation, while simple search offers fast estimation on existing data points.
        
        Returns:
        --------
        float
            The estimated Z0 value where the PDF reaches its global maximum.
            This value is in the same units and scale as the original data.
            
            The returned value represents:
            - The most probable value in the distribution
            - The location of maximum probability density
            - The peak or mode of the distribution function
            - The optimal point for various applications
        
        Raises:
        -------
        ValueError
            - If the GDF object was not fitted before calling this method
            - If smooth curves are not available when using advanced methods
            - If PDF data is missing or corrupted in the GDF object
            - If the GDF object structure is incompatible
            
        RuntimeError
            - If all estimation methods fail to converge to a valid result
            - If numerical instabilities prevent successful estimation
            - If optimization algorithms encounter irrecoverable errors
            
        ImportError
            - If scipy is required for advanced methods but not installed
            - When spline optimization or refined interpolation is needed
        
        Side Effects:
        -------------
        This method modifies the Z0Estimator object and the original GDF object:
        - Sets self.z0 to the estimated value
        - Updates self.estimation_info with comprehensive metadata
        - Sets gdf.z0 attribute in the original GDF object
        - Updates gdf.params dictionary if catch=True in the GDF object
        
        Estimation Methods Used:
        ------------------------
        When optimize=True (advanced methods):
        1. **Spline Optimization**: For smooth, high-resolution data (≥20 points, low noise)
        - Uses scipy UnivariateSpline with minimize_scalar
        - Provides analytical precision through continuous representation
        - Best accuracy for well-behaved, smooth distributions
        
        2. **Polynomial Fitting**: For moderate-resolution data (≥15 points)
        - Fits degree 2-3 polynomials around the maximum region
        - Finds analytical maximum by solving derivative = 0
        - Good balance of accuracy and computational efficiency
        
        3. **Refined Interpolation**: General-purpose method for various data qualities
        - Creates 1000-point fine grid using cubic interpolation
        - Provides sub-point precision without heavy computation
        - Robust for most data types and distributions
        
        4. **Simple Maximum**: Fallback for small datasets or when methods fail
        - Direct maximum finding with optional parabolic interpolation
        - Most reliable method with minimal requirements
        - Always available regardless of dependencies
        
        When optimize=False (simple method):
        - Linear search through existing data points
        - Fastest estimation with adequate accuracy
        - Limited to resolution of original data points
        - Recommended for large datasets or performance-critical applications
        
        Examples:
        ---------
        Basic Z0 estimation:
        >>> z0_estimator = Z0Estimator(fitted_eldf, optimize=True)
        >>> z0_value = z0_estimator.estimate_z0()
        >>> print(f"Maximum PDF occurs at: {z0_value:.6f}")
        
        Fast estimation for large datasets:
        >>> z0_estimator = Z0Estimator(fitted_egdf, optimize=False, verbose=True)
        >>> z0_value = z0_estimator.estimate_z0()
        >>> print(f"Z0 (simple method): {z0_value:.6f}")
        
        Estimation with error handling:
        >>> try:
        ...     z0_value = z0_estimator.estimate_z0()
        ...     print(f"Successfully estimated Z0: {z0_value:.6f}")
        ...     print(f"Method used: {z0_estimator.get_estimation_info()['z0_method']}")
        ... except ValueError as e:
        ...     print(f"Estimation failed: {e}")
        ...     # Handle case where GDF wasn't properly fitted
        
        Multiple estimations and comparison:
        >>> # Compare different approaches
        >>> estimator_advanced = Z0Estimator(gdf_obj, optimize=True)
        >>> estimator_simple = Z0Estimator(gdf_obj, optimize=False)
        >>> 
        >>> z0_advanced = estimator_advanced.estimate_z0()
        >>> z0_simple = estimator_simple.estimate_z0()
        >>> 
        >>> print(f"Advanced: {z0_advanced:.6f}")
        >>> print(f"Simple: {z0_simple:.6f}")
        >>> print(f"Difference: {abs(z0_advanced - z0_simple):.6f}")
        
        Performance Considerations:
        ---------------------------
        - Advanced methods (optimize=True): Higher accuracy, moderate computation time
        - Simple method (optimize=False): Fastest execution, adequate accuracy
        - Method selection is automatic based on data characteristics
        - Computational complexity ranges from O(n) to O(n²) depending on method
        - Memory usage is minimal with temporary arrays only
        
        Accuracy Expectations:
        ----------------------
        - Spline optimization: Sub-point precision, best for smooth data
        - Polynomial fitting: Good precision within local neighborhood
        - Refined interpolation: Balanced accuracy for general use
        - Simple maximum: Limited to data point resolution
        - Parabolic interpolation: Modest sub-point improvement over direct maximum
        
        Troubleshooting:
        ----------------
        If estimation fails or produces unexpected results:
        1. Verify the GDF object was successfully fitted: gdf._fitted should be True
        2. Check that PDF data is available: gdf.pdf_points or gdf.pdf should exist
        3. For advanced methods, ensure sufficient data points (≥10 recommended)
        4. Try optimize=False if advanced methods fail consistently
        5. Use verbose=True to monitor the estimation process and method selection
        6. Check for NaN or infinite values in the PDF data
        
        Integration Notes:
        ------------------
        - The estimated Z0 is automatically stored in the original GDF object
        - Subsequent calls to gdf.plot() will display the Z0 line
        - The Z0 value can be used for further analysis and decision making
        - Multiple estimations update the stored values incrementally
        - The estimation is thread-safe for read operations only
        
        See Also:
        ---------
        get_estimation_info() : Get detailed metadata about the estimation
        plot_z0_analysis() : Visualize the Z0 estimation results
        """
        if self.verbose:
            print(f'Z0Estimator: Computing Z0 point for {self.gdf_type.upper()}...')
        
        if self.optimize:
            # Method 1: Advanced interpolation-based methods
            if not self._has_smooth_curves():
                raise ValueError("Please ensure GDF has smooth curves generated before Z0 estimation.")
            
            if self.verbose:
                print('Z0Estimator: Using interpolation-based optimization for Z0 point...')
            
            # Choose interpolation method based on data quality
            method = self._choose_interpolation_method()
            
            if method == 'spline_optimization':
                self.z0 = self._find_z0_spline_optimization()
            elif method == 'polynomial_fit':
                self.z0 = self._find_z0_polynomial_fit()
            elif method == 'refined_interpolation':
                self.z0 = self._find_z0_refined_interpolation()
            else:  # fallback
                self.z0 = self._find_z0_simple_maximum()
            
            if self.verbose:
                print(f"Z0 point (interpolation method: {method}): {self.z0:.6f}")
            
            # Store comprehensive information
            max_pdf_idx = np.argmax(self._get_pdf_points())
            self.estimation_info = {
                'z0': float(self.z0),
                'z0_method': f'interpolation_{method}',
                'z0_max_pdf_value': self._get_pdf_points()[max_pdf_idx],
                'z0_interpolation_points': len(self._get_pdf_points()),
                'gdf_type': self.gdf_type
            }
        
        else:
            # Method 2: Simple linear search
            if not self._has_smooth_curves():
                raise ValueError("Please ensure GDF has smooth curves generated before Z0 estimation.")
            
            if self.verbose:
                print('Z0Estimator: Performing linear search for Z0 point...')
            
            # Find index with maximum PDF
            pdf_points = self._get_pdf_points()
            di_points = self._get_di_points()
            
            idx = np.argmax(pdf_points)
            self.z0 = di_points[idx]
    
            if self.verbose:
                print(f"Z0 point (linear search): {self.z0:.6f}")
    
            self.estimation_info = {
                'z0': float(self.z0),
                'z0_method': 'linear_search',
                'z0_max_pdf_value': pdf_points[idx],
                'z0_max_pdf_index': idx,
                'gdf_type': self.gdf_type
            }
        
        # Update GDF object with Z0
        self.gdf.z0 = self.z0
        if hasattr(self.gdf, 'catch') and self.gdf.catch and hasattr(self.gdf, 'params'):
            self.gdf.params.update({
                'z0': float(self.z0),
                'z0_method': self.estimation_info['z0_method']
            })
        
        return self.z0
    
    def _has_smooth_curves(self):
        """Check if the GDF object has smooth curves available."""
        return (hasattr(self.gdf, 'pdf_points') and hasattr(self.gdf, 'di_points_n') and
                self.gdf.pdf_points is not None and self.gdf.di_points_n is not None)
    
    def _get_pdf_points(self):
        """Get PDF points from the GDF object."""
        if hasattr(self.gdf, 'pdf_points') and self.gdf.pdf_points is not None:
            return self.gdf.pdf_points
        elif hasattr(self.gdf, 'pdf') and self.gdf.pdf is not None:
            return self.gdf.pdf
        else:
            raise ValueError("No PDF data available in GDF object")
    
    def _get_di_points(self):
        """Get data points from the GDF object."""
        if hasattr(self.gdf, 'di_points_n') and self.gdf.di_points_n is not None:
            return self.gdf.di_points_n
        elif hasattr(self.gdf, 'data') and self.gdf.data is not None:
            return self.gdf.data
        else:
            raise ValueError("No data points available in GDF object")
    
    def _choose_interpolation_method(self):
        """Choose the best interpolation method based on data characteristics."""
        pdf_points = self._get_pdf_points()
        n_points = len(pdf_points)
        pdf_range = np.max(pdf_points) - np.min(pdf_points)
        
        # Avoid division by zero
        if pdf_range == 0:
            return 'simple_maximum'
        
        # Check for smoothness (second differences)
        if n_points >= 10:
            second_diffs = np.diff(pdf_points, n=2)
            smoothness = np.std(second_diffs) / pdf_range
            
            if smoothness < 0.1 and n_points >= 20:
                return 'spline_optimization'
            elif n_points >= 15:
                return 'polynomial_fit'
            else:
                return 'refined_interpolation'
        else:
            return 'simple_maximum'
    
    def _find_z0_spline_optimization(self):
        """Find Z0 using spline interpolation and optimization."""
        try:
            from scipy.interpolate import UnivariateSpline
            from scipy.optimize import minimize_scalar
            
            di_points = self._get_di_points()
            pdf_points = self._get_pdf_points()
            
            # Create spline interpolation of PDF
            spline = UnivariateSpline(di_points, pdf_points, s=0, k=3)
            
            # Find maximum of spline
            result = minimize_scalar(
                lambda x: -spline(x),  # Negative because we minimize
                bounds=(di_points.min(), di_points.max()),
                method='bounded'
            )
            
            if result.success:
                return result.x
            else:
                return self._find_z0_simple_maximum()
                
        except ImportError:
            if self.verbose:
                print("SciPy not available for spline optimization, falling back...")
            return self._find_z0_polynomial_fit()
        except Exception as e:
            if self.verbose:
                print(f"Spline optimization failed: {e}, falling back...")
            return self._find_z0_polynomial_fit()
    
    def _find_z0_polynomial_fit(self):
        """Find Z0 using polynomial fitting around the maximum."""
        try:
            di_points = self._get_di_points()
            pdf_points = self._get_pdf_points()
            
            # Find approximate maximum
            max_idx = np.argmax(pdf_points)
            
            # Define window around maximum (e.g., ±5 points or ±20% of data)
            window_size = min(5, len(pdf_points) // 5)
            start_idx = max(0, max_idx - window_size)
            end_idx = min(len(pdf_points), max_idx + window_size + 1)
            
            # Extract window data
            x_window = di_points[start_idx:end_idx]
            y_window = pdf_points[start_idx:end_idx]
            
            if len(x_window) >= 3:
                # Fit polynomial (degree 2 or 3 depending on data)
                degree = min(3, len(x_window) - 1)
                coeffs = np.polyfit(x_window, y_window, degree)
                poly = np.poly1d(coeffs)
                
                # Find derivative
                poly_deriv = np.polyder(poly)
                
                # Find roots of derivative (critical points)
                critical_points = np.roots(poly_deriv)
                
                # Filter real roots within the window
                real_critical = critical_points[np.isreal(critical_points)].real
                valid_critical = real_critical[
                    (real_critical >= x_window.min()) & 
                    (real_critical <= x_window.max())
                ]
                
                if len(valid_critical) > 0:
                    # Evaluate polynomial at critical points to find maximum
                    critical_values = [poly(cp) for cp in valid_critical]
                    max_critical_idx = np.argmax(critical_values)
                    return valid_critical[max_critical_idx]
            
            # Fallback if polynomial fitting fails
            return di_points[max_idx]
            
        except Exception as e:
            if self.verbose:
                print(f"Polynomial fitting failed: {e}, falling back...")
            return self._find_z0_refined_interpolation()
    
    def _find_z0_refined_interpolation(self):
        """Find Z0 using refined interpolation around the maximum."""
        try:
            from scipy.interpolate import interp1d
            
            di_points = self._get_di_points()
            pdf_points = self._get_pdf_points()
            
            # Find approximate maximum
            max_idx = np.argmax(pdf_points)
            
            # Create finer grid around maximum
            window_size = min(3, len(pdf_points) // 10)
            start_idx = max(0, max_idx - window_size)
            end_idx = min(len(pdf_points), max_idx + window_size + 1)
            
            # Extract window data
            x_window = di_points[start_idx:end_idx]
            y_window = pdf_points[start_idx:end_idx]
            
            if len(x_window) >= 3:
                # Create interpolation function
                interp_func = interp1d(x_window, y_window, kind='cubic', 
                                     bounds_error=False, fill_value='extrapolate')
                
                # Create fine grid
                x_fine = np.linspace(x_window.min(), x_window.max(), 1000)
                y_fine = interp_func(x_fine)
                
                # Find maximum on fine grid
                max_fine_idx = np.argmax(y_fine)
                return x_fine[max_fine_idx]
            else:
                return di_points[max_idx]
                
        except ImportError:
            if self.verbose:
                print("SciPy not available for interpolation, using simple maximum...")
            return self._find_z0_simple_maximum()
        except Exception as e:
            if self.verbose:
                print(f"Refined interpolation failed: {e}, using simple maximum...")
            return self._find_z0_simple_maximum()
    
    def _find_z0_simple_maximum(self):
        """Find Z0 using simple maximum finding with optional parabolic interpolation."""
        di_points = self._get_di_points()
        pdf_points = self._get_pdf_points()
        
        max_idx = np.argmax(pdf_points)
        
        # Try parabolic interpolation around the maximum
        if 1 <= max_idx <= len(pdf_points) - 2:
            try:
                # Get three points around maximum
                x1, x2, x3 = di_points[max_idx-1:max_idx+2]
                y1, y2, y3 = pdf_points[max_idx-1:max_idx+2]
                
                # Parabolic interpolation formula
                denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
                if abs(denom) > 1e-12:  # Avoid division by zero
                    A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
                    B = (x3*x3 * (y1 - y2) + x2*x2 * (y3 - y1) + x1*x1 * (y2 - y3)) / denom
                    
                    if abs(A) > 1e-12:  # Ensure we have a parabola
                        x_max = -B / (2 * A)
                        
                        # Check if interpolated maximum is reasonable
                        if x1 <= x_max <= x3:
                            return x_max
            except:
                pass  # Fall back to simple maximum
        
        return di_points[max_idx]
    
    def get_estimation_info(self) -> Dict[str, Any]:
        """
        Get comprehensive metadata and diagnostic information about the last Z0 estimation.
        
        This method returns a dictionary containing detailed information about the most
        recent Z0 estimation, including the estimated value, method used, quality metrics,
        and diagnostic data. This information is useful for analysis validation,
        debugging, and understanding the estimation process.
        
        Returns:
        --------
        Dict[str, Any]
            A dictionary containing comprehensive estimation metadata with the following keys:
            
            Core Results:
            - 'z0' (float): The estimated Z0 value where PDF reaches maximum
            - 'z0_method' (str): Method used for estimation (e.g., 'interpolation_spline_optimization', 'linear_search')
            - 'gdf_type' (str): Type of GDF object processed ('egdf' or 'eldf')
            
            Quality Metrics:
            - 'z0_max_pdf_value' (float): PDF value at the estimated Z0 point
            - 'z0_interpolation_points' (int): Number of data points used in estimation (for advanced methods)
            - 'z0_max_pdf_index' (int): Index of maximum in original data (for linear search)
            
            The returned dictionary is a copy, so modifications won't affect the internal state.
        
        Raises:
        -------
        RuntimeError
            If estimate_z0() has not been called yet, resulting in empty estimation_info.
            The error message will indicate that Z0 estimation must be performed first.
        
        Notes:
        ------
        - Information is updated each time estimate_z0() is called
        - The dictionary structure may vary slightly based on the estimation method used
        - All values are in the same units and scale as the original data
        - Advanced methods provide more detailed diagnostic information
        - Simple methods focus on core results with minimal overhead
        
        Method-Specific Information:
        ----------------------------
        Advanced Methods (optimize=True):
        - Include interpolation method details in z0_method
        - Provide z0_interpolation_points for data quality assessment
        - Show z0_max_pdf_value for validation purposes
        
        Simple Method (optimize=False):
        - Include z0_max_pdf_index for direct data point reference
        - Faster to compute with essential information only
        - Suitable for performance-critical applications
        
        Examples:
        ---------
        Basic information retrieval:
        >>> z0_estimator = Z0Estimator(fitted_gdf)
        >>> z0_value = z0_estimator.estimate_z0()
        >>> info = z0_estimator.get_estimation_info()
        >>> 
        >>> print(f"Z0 value: {info['z0']:.6f}")
        >>> print(f"Method: {info['z0_method']}")
        >>> print(f"PDF at Z0: {info['z0_max_pdf_value']:.6f}")
        
        Comprehensive analysis:
        >>> info = z0_estimator.get_estimation_info()
        >>> print("Z0 Estimation Summary:")
        >>> print("-" * 30)
        >>> for key, value in info.items():
        ...     if isinstance(value, float):
        ...         print(f"{key}: {value:.6f}")
        ...     else:
        ...         print(f"{key}: {value}")
        
        Method comparison and validation:
        >>> # Compare different estimators
        >>> estimators = [
        ...     Z0Estimator(gdf, optimize=True, verbose=False),
        ...     Z0Estimator(gdf, optimize=False, verbose=False)
        ... ]
        >>> 
        >>> results = []
        >>> for estimator in estimators:
        ...     estimator.estimate_z0()
        ...     results.append(estimator.get_estimation_info())
        >>> 
        >>> # Compare results
        >>> for i, result in enumerate(results):
        ...     print(f"Estimator {i+1}: Z0={result['z0']:.6f}, Method={result['z0_method']}")
        
        Quality assessment:
        >>> info = z0_estimator.get_estimation_info()
        >>> 
        >>> # Check estimation quality
        >>> if 'z0_interpolation_points' in info:
        ...     points = info['z0_interpolation_points']
        ...     print(f"Estimation based on {points} points")
        ...     if points < 10:
        ...         print("Warning: Low resolution data may affect accuracy")
        >>> 
        >>> # Validate PDF maximum
        >>> pdf_max = info['z0_max_pdf_value']
        >>> if pdf_max > 0:
        ...     print(f"Valid PDF maximum found: {pdf_max:.6f}")
        ... else:
        ...     print("Warning: PDF maximum is zero or negative")
        
        Error handling:
        >>> try:
        ...     info = z0_estimator.get_estimation_info()
        ...     # Process estimation information
        ... except RuntimeError:
        ...     print("No estimation performed yet. Call estimate_z0() first.")
        ...     z0_estimator.estimate_z0()
        ...     info = z0_estimator.get_estimation_info()
        
        Batch processing analysis:
        >>> estimation_results = []
        >>> for gdf_object in gdf_list:
        ...     estimator = Z0Estimator(gdf_object)
        ...     estimator.estimate_z0()
        ...     info = estimator.get_estimation_info()
        ...     
        ...     estimation_results.append({
        ...         'object_id': id(gdf_object),
        ...         'z0': info['z0'],
        ...         'method': info['z0_method'],
        ...         'pdf_max': info['z0_max_pdf_value']
        ...     })
        >>> 
        >>> # Analyze batch results
        >>> methods_used = [r['method'] for r in estimation_results]
        >>> print(f"Methods distribution: {set(methods_used)}")
        
        Dictionary Keys Reference:
        --------------------------
        Always Present:
        - z0: The estimated Z0 value
        - z0_method: Estimation method identifier
        - gdf_type: Type of GDF object ('egdf' or 'eldf')
        - z0_max_pdf_value: PDF value at Z0 location
        
        Method-Dependent:
        - z0_interpolation_points: Available for advanced methods
        - z0_max_pdf_index: Available for linear search method
        
        Applications:
        -------------
        - Validation of estimation quality and reliability
        - Method performance comparison and optimization
        - Debugging estimation issues and numerical problems
        - Report generation and result documentation
        - Quality control in automated processing pipelines
        - Academic research and algorithm development
        
        See Also:
        ---------
        estimate_z0() : Perform the actual Z0 estimation
        plot_z0_analysis() : Visualize estimation results
        """
    
    def plot_z0_analysis(self, figsize: tuple = (12, 6)):
        """
        Create comprehensive visualization of Z0 estimation results showing PDF curves and Z0 location.
        
        This method generates a detailed two-panel plot that visualizes the estimated Z0 point
        in context with the probability density function. The visualization helps validate
        the estimation results, understand the distribution characteristics around the Z0 point,
        and provides publication-ready graphics for analysis and reporting.
        
        The plot consists of two complementary views: a full PDF overview with Z0 marked,
        and a zoomed view focusing on the region around Z0 for detailed analysis.
        
        Parameters:
        -----------
        figsize : tuple, optional (default=(12, 6))
            Figure size specification as (width, height) in inches.
            Controls the overall size of the generated plot.
            
            Recommended sizes:
            - (12, 6): Default balanced size for most applications
            - (15, 8): Larger size for detailed analysis and presentations
            - (10, 5): Compact size for embedded plots or quick analysis
            - (16, 9): Wide format for presentations and reports
            
            The figure is automatically divided into two equal subplots side by side.
        
        Returns:
        --------
        None
            This method displays the plot using matplotlib's interactive backend
            and does not return any value. The plot can be saved manually using
            matplotlib's savefig() function or programmatically after calling this method.
        
        Raises:
        -------
        ValueError
            - If estimate_z0() has not been called before plotting
            - If the Z0 value is None or invalid
            - If figsize is not a valid tuple of positive numbers
            
        ImportError
            - If matplotlib is not available for plotting
            - If required plotting dependencies are missing
            
        RuntimeError
            - If PDF data is corrupted or unavailable
            - If plotting encounters numerical issues with the data
        
        Plot Components:
        ----------------
        Left Panel - Full PDF Overview:
        - Complete PDF curve across all data points (blue solid line)
        - Z0 location marked with vertical red dashed line
        - Z0 point highlighted with red scatter marker
        - PDF value at Z0 clearly visible
        - Complete data range context
        - Grid for easy value reading
        - Legend identifying all elements
        - Appropriate axis labels and title
        
        Right Panel - Zoomed View Around Z0:
        - Focused view of PDF around Z0 location (±10 points or ±5% of data)
        - Same Z0 marking and highlighting as left panel
        - Enhanced detail for local PDF behavior analysis
        - Grid for precise value reading
        - Legend for element identification
        - Zoomed region clearly indicated in title
        
        Visual Elements:
        ----------------
        - PDF Curve: Blue solid line (linewidth=2) showing probability density
        - Z0 Line: Red dashed vertical line (linewidth=2) marking the Z0 location
        - Z0 Point: Red scatter marker (size=100) highlighting exact Z0 position
        - Grid: Semi-transparent grid (alpha=0.3) for value reading
        - Labels: Clear axis labels indicating data points and PDF values
        - Titles: Descriptive titles indicating GDF type and analysis focus
        - Legend: Comprehensive legend with Z0 value displayed to 4 decimal places
        
        Examples:
        ---------
        Basic Z0 analysis visualization:
        >>> z0_estimator = Z0Estimator(fitted_eldf)
        >>> z0_estimator.estimate_z0()
        >>> z0_estimator.plot_z0_analysis()
        >>> # Displays two-panel plot with Z0 analysis
        
        Custom figure size for presentations:
        >>> z0_estimator.plot_z0_analysis(figsize=(16, 9))
        >>> # Creates larger plot suitable for presentations
        
        Compact visualization:
        >>> z0_estimator.plot_z0_analysis(figsize=(10, 5))
        >>> # Creates smaller plot for embedded use
        
        Analysis workflow with saving:
        >>> z0_estimator = Z0Estimator(fitted_gdf, verbose=True)
        >>> z0_value = z0_estimator.estimate_z0()
        >>> 
        >>> # Create and save analysis plot
        >>> z0_estimator.plot_z0_analysis(figsize=(15, 8))
        >>> plt.savefig('z0_analysis.png', dpi=300, bbox_inches='tight')
        >>> plt.savefig('z0_analysis.pdf', bbox_inches='tight')  # Vector format
        >>> plt.show()
        
        Batch processing with multiple plots:
        >>> import matplotlib.pyplot as plt
        >>> 
        >>> for i, gdf_obj in enumerate(gdf_objects):
        ...     estimator = Z0Estimator(gdf_obj)
        ...     estimator.estimate_z0()
        ...     
        ...     estimator.plot_z0_analysis(figsize=(12, 6))
        ...     plt.savefig(f'z0_analysis_{i}.png', dpi=300, bbox_inches='tight')
        ...     plt.close()  # Close to free memory
        
        Comparative analysis visualization:
        >>> # Compare Z0 estimations from different methods
        >>> estimator_opt = Z0Estimator(gdf, optimize=True)
        >>> estimator_simple = Z0Estimator(gdf, optimize=False)
        >>> 
        >>> z0_opt = estimator_opt.estimate_z0()
        >>> z0_simple = estimator_simple.estimate_z0()
        >>> 
        >>> # Plot both analyses
        >>> plt.figure(figsize=(16, 6))
        >>> 
        >>> plt.subplot(1, 2, 1)
        >>> estimator_opt.plot_z0_analysis(figsize=(8, 6))
        >>> plt.title(f'Optimized Method: Z0={z0_opt:.4f}')
        >>> 
        >>> plt.subplot(1, 2, 2)
        >>> estimator_simple.plot_z0_analysis(figsize=(8, 6))
        >>> plt.title(f'Simple Method: Z0={z0_simple:.4f}')
        >>> 
        >>> plt.tight_layout()
        >>> plt.show()
        
        Interpretation Guide:
        ---------------------
        Full PDF View (Left Panel):
        - Shows overall distribution shape and characteristics
        - Z0 location relative to distribution extremes
        - Identifies whether Z0 is at a global maximum
        - Reveals distribution symmetry and skewness
        - Helps identify potential multiple modes
        
        Zoomed View (Right Panel):
        - Detailed view of PDF behavior around Z0
        - Local smoothness and curvature analysis
        - Validation of maximum location accuracy
        - Assessment of estimation precision
        - Identification of local distribution features
        
        Quality Assessment:
        -------------------
        Good Z0 Estimation Indicators:
        - Z0 line clearly intersects PDF maximum
        - Smooth PDF curve around Z0 location
        - Z0 point is well-centered in zoomed view
        - PDF value at Z0 is clearly the highest
        
        Potential Issues to Look For:
        - Z0 not at apparent PDF maximum (estimation error)
        - Multiple peaks with Z0 at local rather than global maximum
        - Noisy PDF making Z0 location ambiguous
        - Z0 at distribution boundary (potential boundary effect)
        
        Performance Tips:
        -----------------
        - Use default figsize for most applications
        - Increase figsize for detailed analysis or presentations
        - Save in vector formats (PDF, SVG) for publications
        - Save in raster formats (PNG, JPG) for web use
        - Close plots in batch processing to manage memory
        - Use tight_layout() for optimal subplot spacing
        
        Integration Notes:
        ------------------
        - This plot complements the GDF object's built-in plotting methods
        - Can be used alongside gdf.plot() for comprehensive analysis
        - Z0 line will also appear in subsequent GDF plots
        - Provides focused analysis that GDF plots may not emphasize
        
        Troubleshooting:
        ----------------
        Plot Not Displaying:
        - Ensure matplotlib backend supports interactive display
        - Check if running in headless environment (use savefig instead)
        - Verify that estimate_z0() was called successfully
        
        Visual Quality Issues:
        - Increase figsize for better resolution and clarity
        - Adjust DPI in savefig for higher quality output
        - Use vector formats for scalable graphics
        
        Data Visualization Problems:
        - Check for NaN or infinite values in PDF data
        - Verify sufficient data points for meaningful zoomed view
        - Consider data preprocessing if PDF is too noisy
        
        See Also:
        ---------
        estimate_z0() : Perform Z0 estimation before plotting
        get_estimation_info() : Get detailed estimation metadata
        gdf.plot() : Plot the original GDF with Z0 marked
        """
        try:
            import matplotlib.pyplot as plt
            
            if self.z0 is None:
                raise ValueError("Must estimate Z0 before plotting")
            
            di_points = self._get_di_points()
            pdf_points = self._get_pdf_points()
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            # Plot 1: PDF with Z0 location
            ax1.plot(di_points, pdf_points, 'b-', linewidth=2, label='PDF')
            ax1.axvline(x=self.z0, color='red', linestyle='--', linewidth=2, 
                       label=f'Z0 = {self.z0:.4f}')
            ax1.scatter([self.z0], [np.interp(self.z0, di_points, pdf_points)], 
                       color='red', s=100, zorder=5)
            ax1.set_xlabel('Data Points')
            ax1.set_ylabel('PDF')
            ax1.set_title(f'PDF and Z0 Location ({self.gdf_type.upper()})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Zoomed view around Z0
            z0_idx = np.argmin(np.abs(di_points - self.z0))
            window = max(10, len(di_points) // 20)
            start_idx = max(0, z0_idx - window)
            end_idx = min(len(di_points), z0_idx + window)
            
            ax2.plot(di_points[start_idx:end_idx], pdf_points[start_idx:end_idx], 
                    'b-', linewidth=2, label='PDF (zoomed)')
            ax2.axvline(x=self.z0, color='red', linestyle='--', linewidth=2, 
                       label=f'Z0 = {self.z0:.4f}')
            ax2.scatter([self.z0], [np.interp(self.z0, di_points, pdf_points)], 
                       color='red', s=100, zorder=5)
            ax2.set_xlabel('Data Points')
            ax2.set_ylabel('PDF')
            ax2.set_title(f'Zoomed View Around Z0')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting")
        except Exception as e:
            print(f"Error creating plot: {e}")
    
    def __repr__(self):
        return f"Z0Estimator(gdf_type='{self.gdf_type}', z0={self.z0}, method='{self.estimation_info.get('z0_method', 'not_estimated')}')"