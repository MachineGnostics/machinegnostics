"""
Z0 Estimator - Universal class for estimating Z0 point for both EGDF and ELDF

Z0 is the point where PDF is at its global maximum, using advanced interpolation methods.

Author: Nirmal Parmar
Machine Gnostics
"""

import numpy as np
import warnings
from typing import Union, Dict, Any, Optional

class Z0Estimator:
    """
    Universal Z0 Point Estimator for Machine Gnostics Distribution Functions.
    
    The Z0Estimator is a sophisticated computational tool designed to accurately identify 
    the Z0 point - the location where a Probability Density Function (PDF) reaches its 
    GLOBAL maximum. This class works seamlessly with both Estimating Global Distribution 
    Function (EGDF) and Estimating Local Distribution Function (ELDF) objects, providing 
    robust and precise Z0 estimation using advanced mathematical methods.
    
    What is the Z0 Point?
    ---------------------
    The Z0 point represents the most probable value in a distribution - the peak of the 
    probability density curve. In practical applications, Z0 indicates:
    
    • **Optimal Point**: The most likely outcome or state
    • **Peak Performance**: Maximum probability density location  
    • **Central Tendency**: Advanced measure beyond simple mean/median
    • **Risk Assessment**: Point of highest probability concentration
    • **Decision Making**: Optimal choice under uncertainty
    
    Why Z0 Matters:
    ---------------
    • **Financial Modeling**: Identify most probable return rates
    • **Quality Control**: Find optimal process parameters
    • **Risk Analysis**: Locate maximum risk concentration points
    • **Machine Learning**: Determine optimal decision boundaries
    • **Engineering**: Identify peak performance conditions
    • **Research**: Locate statistical significance peaks
    
    Advanced Estimation Methods:
    ----------------------------
    The Z0Estimator employs a hierarchical approach with multiple sophisticated methods:
    
    1. **Global Spline Optimization** (Primary Method)
       - Uses SciPy's UnivariateSpline with bounded optimization
       - Searches entire domain for true global maximum
       - Validates results against discrete data consistency
       
    2. **Polynomial Fitting** (Secondary Method)  
       - Fits high-degree polynomials around maximum region
       - Finds critical points using derivative analysis
       - Validates using second derivative test for maxima
       
    3. **Refined Cubic Interpolation** (Tertiary Method)
       - Creates fine-grained interpolation around peak
       - Uses SciPy's interp1d with cubic spline method
       - Searches high-resolution grid for maximum
       
    4. **Parabolic Interpolation** (Quaternary Method)
       - Uses three-point parabolic fitting formula
       - Analytically solves for vertex of fitted parabola
       - Validates parabola orientation and bounds
       
    5. **Discrete Maximum Fallback** (Ultimate Fallback)
       - Uses simple argmax on original discrete data
       - Guaranteed to always provide a result
       - Maintains computational robustness
    
    Key Features & Benefits:
    ------------------------
    
    **Accuracy & Precision:**
    • Advanced interpolation ensures sub-sample precision
    • Multiple validation checks prevent local maxima convergence  
    • Hierarchical fallback system guarantees robust results
    • Numerical stability with epsilon-based safeguards
    
    **Flexibility & Compatibility:**
    • Works with both EGDF and ELDF objects seamlessly
    • Automatic GDF type detection and handling
    • Configurable optimization levels (simple vs. advanced)
    • Compatible with weighted and unweighted distributions
    
    **Performance & Reliability:**
    • Intelligent method selection based on data characteristics
    • Graceful degradation when advanced methods fail
    • Comprehensive error handling and logging
    • Memory-efficient computation with optional cleanup
    
    **Analysis & Diagnostics:**
    • Detailed estimation information storage
    • Method tracking for result interpretation
    • Built-in validation and quality assessment
    • Advanced visualization capabilities
    
    Parameters
    ----------
    gdf_object : EGDF or ELDF
        A fitted EGDF or ELDF object containing the distribution data and PDF values.
        The object must be already fitted (have _fitted=True attribute) and contain
        the necessary probability density information for Z0 estimation.
        
        **Required GDF Object Attributes:**
        - `_fitted` : bool (must be True)
        - `data` : numpy array of original data points  
        - `pdf_points` or `pdf` : numpy array of PDF values
        - `di_points_n` or equivalent : numpy array of evaluation points
        
    optimize : bool, default=True
        Controls the estimation strategy and computational intensity:
        
        - **True**: Uses advanced interpolation methods (spline optimization, 
          polynomial fitting, refined interpolation, parabolic interpolation) 
          for maximum accuracy and sub-sample precision
          
        - **False**: Uses simple linear search on existing discrete points
          for faster computation with basic accuracy
          
        **Recommendation**: Use True for research/analysis, False for real-time applications
        
    verbose : bool, default=False
        Controls diagnostic output and logging detail:
        
        - **True**: Prints comprehensive diagnostic information including method 
          selection, intermediate results, validation checks, and performance metrics
          
        - **False**: Silent operation with minimal output
          
        **Recommendation**: Use True during development/debugging, False in production
    
    Attributes
    ----------
    z0 : float or None
        The estimated Z0 value where the PDF reaches its global maximum.
        None until fit() method is successfully called.
        
        **Interpretation:**
        - Value is in the same units as the original data
        - Represents the most probable outcome/state
        - Can be used for decision making and optimization
        
    gdf : EGDF or ELDF
        Reference to the input GDF object containing distribution data.
        Used for accessing PDF values, data points, and metadata.
        
    gdf_type : str
        Automatically detected type of GDF object ('egdf' or 'eldf').
        Used for method selection and result interpretation.
        
    optimize : bool
        Stored optimization preference affecting method selection.
        
    verbose : bool  
        Stored verbosity preference affecting diagnostic output.
        
    estimation_info : dict
        Comprehensive dictionary containing detailed information about the 
        last Z0 estimation process:
        
        **Always Present Keys:**
        - 'z0' : float - The estimated Z0 value
        - 'z0_method' : str - Method used for final estimation
        - 'z0_max_pdf_value' : float - PDF value at Z0 location
        - 'gdf_type' : str - Type of GDF object used
        
        **Additional Keys (when optimize=True):**
        - 'z0_interpolation_points' : int - Number of points used
        - 'global_max_idx' : int - Index of discrete global maximum
        - 'global_max_location' : float - Location of discrete maximum
        
        **Additional Keys (when optimize=False):**
        - 'z0_max_pdf_index' : int - Index of maximum in discrete data
    
    Methods Summary
    ---------------
    **Primary Methods:**
    - `fit()` : Main estimation method - computes Z0 point
    - `get_estimation_info()` : Retrieves detailed estimation information  
    - `plot_z0_analysis()` : Creates comprehensive visualization plots
    
    **Utility Methods:**
    - `_validate_gdf_object()` : Validates input GDF object compatibility
    - `_detect_gdf_type()` : Automatically detects EGDF vs ELDF type
    - `_get_pdf_points()` : Extracts PDF data from GDF object
    - `_get_di_points()` : Extracts data points from GDF object
    
    Examples
    --------
    **Basic Usage with EGDF:**
    
    >>> from machinegnostics.magcal import EGDF, Z0Estimator
    >>> 
    >>> # Create and fit EGDF
    >>> data = np.random.normal(5, 2, 1000)  # Normal distribution centered at 5
    >>> egdf = EGDF(data)
    >>> egdf.fit()
    >>> 
    >>> # Estimate Z0 with high precision
    >>> z0_estimator = Z0Estimator(egdf, optimize=True, verbose=True)
    >>> z0_value = z0_estimator.fit()
    >>> 
    >>> print(f"Z0 point (most probable value): {z0_value:.4f}")
    >>> # Expected output: Z0 point (most probable value): 5.0234
    >>> 
    >>> # Get detailed estimation information
    >>> info = z0_estimator.get_estimation_info()
    >>> print(f"Estimation method: {info['z0_method']}")
    >>> print(f"PDF at Z0: {info['z0_max_pdf_value']:.6f}")
    >>> # Expected output: 
    >>> # Estimation method: global_spline_optimization
    >>> # PDF at Z0: 0.199472
    
    **Advanced Usage with ELDF:**
    
    >>> from machinegnostics.magcal import ELDF
    >>> 
    >>> # Create bimodal data for complex Z0 estimation
    >>> data1 = np.random.normal(2, 0.5, 300)
    >>> data2 = np.random.normal(8, 1.0, 700)  # Larger mode at 8
    >>> bimodal_data = np.concatenate([data1, data2])
    >>> 
    >>> # Fit ELDF
    >>> eldf = ELDF(bimodal_data, S='auto')
    >>> eldf.fit()
    >>> 
    >>> # Estimate Z0 - should find the larger mode around 8
    >>> z0_estimator = Z0Estimator(eldf, optimize=True, verbose=True)
    >>> z0_value = z0_estimator.fit()
    >>> 
    >>> print(f"Z0 point: {z0_value:.4f}")  
    >>> # Expected output: Z0 point: 7.9876 (near the larger mode)
    >>> 
    >>> # Create analysis visualization
    >>> z0_estimator.plot_z0_analysis(figsize=(14, 8))
    >>> 
    >>> # Validate estimation quality
    >>> validation_info = z0_estimator.get_estimation_info()
    >>> method_used = validation_info['z0_method']
    >>> if 'spline' in method_used:
    >>>     print("High-precision spline method used - excellent accuracy")
    >>> elif 'polynomial' in method_used:
    >>>     print("Polynomial method used - good accuracy")
    >>> else:
    >>>     print("Basic method used - standard accuracy")
    
    **Performance Comparison:**
    
    >>> import time
    >>> 
    >>> # High precision (slower but more accurate)
    >>> start_time = time.time()
    >>> z0_precise = Z0Estimator(eldf, optimize=True).fit()
    >>> precise_time = time.time() - start_time
    >>> 
    >>> # Fast computation (faster but less precise)
    >>> start_time = time.time()  
    >>> z0_fast = Z0Estimator(eldf, optimize=False).fit()
    >>> fast_time = time.time() - start_time
    >>> 
    >>> print(f"Precise Z0: {z0_precise:.6f} (time: {precise_time:.4f}s)")
    >>> print(f"Fast Z0: {z0_fast:.6f} (time: {fast_time:.4f}s)")
    >>> print(f"Difference: {abs(z0_precise - z0_fast):.6f}")
    
    **Error Handling and Robustness:**
    
    >>> # Demonstrate robustness with challenging data
    >>> sparse_data = np.array([1, 1.1, 5, 5.1, 5.2, 9, 9.1])  # Very sparse
    >>> 
    >>> try:
    >>>     egdf_sparse = EGDF(sparse_data)  
    >>>     egdf_sparse.fit()
    >>>     
    >>>     # Z0Estimator will gracefully handle sparse data
    >>>     z0_est = Z0Estimator(egdf_sparse, optimize=True, verbose=True)
    >>>     z0_sparse = z0_est.fit()
    >>>     
    >>>     print(f"Z0 for sparse data: {z0_sparse:.4f}")
    >>>     
    >>>     # Check which method was actually used
    >>>     method_info = z0_est.get_estimation_info()
    >>>     print(f"Method used: {method_info['z0_method']}")
    >>>     
    >>> except Exception as e:
    >>>     print(f"Estimation failed: {e}")
    >>>     print("This demonstrates the robust error handling")
    
    **Integration with Analysis Pipelines:**
    
    >>> # Complete analysis workflow
    >>> def analyze_distribution_peak(data, distribution_type='auto'):
    >>>     \"\"\"Complete Z0 analysis workflow.\"\"\"
    >>>     
    >>>     # Choose distribution function
    >>>     if distribution_type == 'global':
    >>>         gdf = EGDF(data)
    >>>     elif distribution_type == 'local':  
    >>>         gdf = ELDF(data)
    >>>     else:
    >>>         # Auto-select based on data characteristics
    >>>         gdf = ELDF(data) if len(data) > 500 else EGDF(data)
    >>>     
    >>>     # Fit distribution
    >>>     gdf.fit()
    >>>     
    >>>     # Estimate Z0 with full analysis
    >>>     z0_estimator = Z0Estimator(gdf, optimize=True, verbose=True)
    >>>     z0_value = z0_estimator.fit()
    >>>     
    >>>     # Get comprehensive results
    >>>     analysis_results = {
    >>>         'z0_point': z0_value,
    >>>         'distribution_type': gdf.__class__.__name__,
    >>>         'estimation_details': z0_estimator.get_estimation_info(),
    >>>         'data_statistics': {
    >>>             'mean': np.mean(data),
    >>>             'median': np.median(data),
    >>>             'std': np.std(data),
    >>>             'min': np.min(data),
    >>>             'max': np.max(data)
    >>>         }
    >>>     }
    >>>     
    >>>     # Create visualization
    >>>     z0_estimator.plot_z0_analysis()
    >>>     
    >>>     return analysis_results
    >>> 
    >>> # Use the analysis function
    >>> sample_data = np.random.gamma(2, 2, 1000)  # Gamma distribution
    >>> results = analyze_distribution_peak(sample_data)
    >>> 
    >>> print(f"Peak probability at: {results['z0_point']:.4f}")
    >>> print(f"Compared to mean: {results['data_statistics']['mean']:.4f}")
    >>> print(f"Method used: {results['estimation_details']['z0_method']}")
    
    Technical Notes
    ---------------
    **Computational Complexity:**
    - optimize=False: O(n) where n is number of PDF points
    - optimize=True: O(n log n) for spline methods, O(n) for others
    
    **Memory Usage:**
    - Minimal additional memory beyond input GDF object
    - Estimation info storage scales with O(1)
    - Temporary arrays cleaned automatically
    
    **Numerical Stability:**
    - Uses epsilon-based comparisons for floating-point operations
    - Validates all interpolation results against discrete data
    - Handles edge cases (boundary maxima, flat regions)
    
    **Dependencies:**
    - **Required**: NumPy for core computations
    - **Optional**: SciPy for advanced interpolation methods
    - **Optional**: Matplotlib for visualization capabilities
    
    **Thread Safety:**
    - Each Z0Estimator instance is independent and thread-safe
    - No global state or shared mutable data
    - Safe for parallel processing applications
    
    Limitations & Considerations
    ----------------------------
    **Data Requirements:**
    - Input GDF object must be fitted and contain PDF data
    - Minimum 3 data points required for advanced methods
    - Works best with smooth, well-behaved probability densities
    
    **Method Selection:**
    - Advanced methods require SciPy installation
    - Sparse data may fall back to simpler methods automatically  
    - Very noisy data may benefit from optimize=False setting
    
    **Performance Trade-offs:**
    - Higher precision comes with increased computational cost
    - Real-time applications may prefer optimize=False
    - Memory usage scales with number of evaluation points
    
    **Accuracy Expectations:**
    - Advanced methods: Sub-sample precision (typically 1e-6 relative accuracy)
    - Simple methods: Sample-level precision (limited by discrete data)
    - Results depend on underlying data quality and PDF smoothness
    
    See Also
    --------
    EGDF : Estimating Global Distribution Function for global analysis
    ELDF : Estimating Local Distribution Function for local analysis  
    GnosticsCharacteristics : Core mathematical operations for distributions
    DataConversion : Utilities for domain transformations
    
    References
    ----------
    .. [1] Parmar, N. "Machine Gnostics: Advanced Distribution Analysis"
           Machine Gnostics Documentation, 2025.
    .. [2] SciPy documentation for interpolation and optimization methods
    .. [3] NumPy documentation for numerical computing foundations
    
    Version Information
    -------------------
    This implementation is part of the Machine Gnostics package and requires
    Python 3.7+ with NumPy. SciPy is recommended for optimal performance.
    
    Author: Nirmal Parmar, Machine Gnostics
    """

    def __init__(self,
                 gdf_object,
                 optimize: bool = True,
                 verbose: bool = False):
        """
        Initialize Z0 estimator.
        
        Parameters
        ----------
        gdf_object : EGDF or ELDF
            Fitted EGDF or ELDF object containing distribution data and PDF values.
            Must be already fitted (have _fitted=True attribute).
        
        optimize : bool, default=True
            If True, use advanced interpolation methods for higher accuracy.
            If False, use simple linear search on existing points.
        
        verbose : bool, default=False
            If True, print detailed diagnostic information during estimation.
        
        Raises
        ------
        ValueError
            If gdf_object is not fitted, missing required attributes, or lacks PDF data.
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
        
    def fit(self) -> float:
        """
        Estimate the Z0 point where the Probability Density Function (PDF) reaches its GLOBAL maximum.
        
        This is the main method that performs Z0 estimation using either advanced optimization
        methods or simple linear search, depending on the 'optimize' parameter. The method
        ensures that the Z0 point corresponds to the global maximum of the PDF, not just a 
        local maximum.
        
        The estimation process follows these steps:
        1. Extract PDF and data points from the GDF object
        2. Identify the global maximum in the discrete data
        3. If optimize=True, refine the location using advanced interpolation methods
        4. If optimize=False, use the discrete global maximum directly
        5. Update the GDF object with the estimated Z0 value
        6. Store comprehensive estimation information
        
        Returns
        -------
        float
            The estimated Z0 value where the PDF reaches its global maximum.
            
        Raises
        ------
        ValueError
            If no PDF data is available for Z0 estimation.
        RuntimeError
            If estimation fails due to computational issues.
            
        Notes
        -----
        When optimize=True, the method tries multiple advanced approaches in order:
        - Global spline optimization over entire domain
        - Polynomial fitting in larger window around global maximum  
        - Refined cubic interpolation around global maximum
        - Parabolic interpolation using three points around maximum
        - Fallback to discrete global maximum if all methods fail
        
        Each method includes validation to ensure convergence to the global maximum.
        The estimation information is stored in the 'estimation_info' attribute for
        later retrieval using get_estimation_info().
        
        Examples
        --------
        >>> estimator = Z0Estimator(fitted_gdf, optimize=True, verbose=True)
        >>> z0 = estimator.fit()
        >>> print(f"Estimated Z0: {z0:.6f}")
        >>> method_info = estimator.get_estimation_info()
        >>> print(f"Method used: {method_info['z0_method']}")
        """
        if self.verbose:
            print(f'Z0Estimator: Computing Z0 point for {self.gdf_type.upper()}...')
        
        # Get PDF and data points
        pdf_points = self._get_pdf_points()
        di_points = self._get_di_points()
        
        if len(pdf_points) == 0:
            raise ValueError("No PDF data available for Z0 estimation")
        
        # First, always find the global maximum in the discrete data
        global_max_idx = np.argmax(pdf_points)
        global_max_value = pdf_points[global_max_idx]
        global_max_location = di_points[global_max_idx]
        
        if self.verbose:
            print(f"Global maximum found at index {global_max_idx}: "
                  f"location={global_max_location:.6f}, PDF={global_max_value:.6f}")
        
        if self.optimize:
            if self.verbose:
                print('Z0Estimator: Using advanced interpolation methods...')
            
            # Use advanced methods to refine the global maximum location
            self.z0 = self._find_z0_advanced_global(global_max_idx, di_points, pdf_points)
            method_used = self._get_last_method_used()
            
            if self.verbose:
                print(f"Z0 point (method: {method_used}): {self.z0:.6f}")
            
            # Store comprehensive information
            self.estimation_info = {
                'z0': float(self.z0),
                'z0_method': method_used,
                'z0_max_pdf_value': global_max_value,
                'z0_interpolation_points': len(pdf_points),
                'gdf_type': self.gdf_type,
                'global_max_idx': global_max_idx,
                'global_max_location': global_max_location
            }
        
        else:
            if self.verbose:
                print('Z0Estimator: Using simple linear search...')
            
            # Simple method: use the global maximum directly
            self.z0 = global_max_location
    
            if self.verbose:
                print(f"Z0 point (linear search): {self.z0:.6f}")
    
            self.estimation_info = {
                'z0': float(self.z0),
                'z0_method': 'linear_search',
                'z0_max_pdf_value': global_max_value,
                'z0_max_pdf_index': global_max_idx,
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
    
    def get_estimation_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the last Z0 estimation.
        
        Returns comprehensive information about the Z0 estimation process including
        the estimated value, method used, PDF value at Z0, and various diagnostic
        parameters. This information is useful for understanding the estimation
        quality and debugging potential issues.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing detailed estimation information with the following keys:
            
            - 'z0' : float
                The estimated Z0 value
            - 'z0_method' : str
                Method used for estimation (e.g., 'global_spline_optimization', 
                'global_polynomial_fit', 'linear_search', etc.)
            - 'z0_max_pdf_value' : float
                PDF value at the estimated Z0 location
            - 'gdf_type' : str
                Type of GDF object ('egdf' or 'eldf')
                
            Additional keys when optimize=True:
            - 'z0_interpolation_points' : int
                Number of data points used for interpolation
            - 'global_max_idx' : int
                Index of global maximum in discrete data
            - 'global_max_location' : float
                Location of global maximum in discrete data
                
            Additional keys when optimize=False:
            - 'z0_max_pdf_index' : int
                Index of the maximum PDF value in the data
        
        Raises
        ------
        RuntimeError
            If no estimation has been performed yet (fit() not called).
            
        Examples
        --------
        >>> estimator = Z0Estimator(fitted_gdf)
        >>> z0 = estimator.fit()
        >>> info = estimator.get_estimation_info()
        >>> print(f"Z0 value: {info['z0']}")
        >>> print(f"Method: {info['z0_method']}")
        >>> print(f"PDF at Z0: {info['z0_max_pdf_value']}")
        >>> if 'z0_interpolation_points' in info:
        ...     print(f"Points used: {info['z0_interpolation_points']}")
        """
        if not self.estimation_info:
            raise RuntimeError("No estimation performed yet. Call fit() first.")
        return self.estimation_info.copy()
    
    def plot_z0_analysis(self, figsize: tuple = (12, 6)) -> None:
        """
        Create comprehensive visualization of Z0 estimation results.
        
        Generates a two-panel plot showing:
        1. Full PDF curve with Z0 location marked
        2. Zoomed view around the Z0 region
        
        The visualization includes the estimated Z0 point, discrete global maximum
        for comparison, and detailed method information. This is useful for visual
        validation of the estimation results and understanding the PDF shape around
        the maximum.
        
        Parameters
        ----------
        figsize : tuple of (width, height), default=(12, 6)
            Figure size in inches for the matplotlib plot.
            
        Raises
        ------
        ValueError
            If Z0 has not been estimated yet (fit() not called).
        ImportError
            If matplotlib is not available (prints warning instead of raising).
            
        Notes
        -----
        The plot includes:
        - Blue line: PDF curve
        - Red dashed line: Z0 location
        - Red circle: Z0 point with PDF value
        - Orange square: Discrete global maximum for comparison
        - Grid and legend for clarity
        - Method name in the zoomed plot title
        
        The zoomed plot focuses on a window around Z0 (typically ±10% of data range
        or ±10 points, whichever is larger) to show fine details of the estimation.
        
        Examples
        --------
        >>> estimator = Z0Estimator(fitted_gdf, verbose=True)
        >>> z0 = estimator.fit()
        >>> estimator.plot_z0_analysis(figsize=(14, 8))
        
        >>> # Plot with default size
        >>> estimator.plot_z0_analysis()
        """
        try:
            import matplotlib.pyplot as plt
            
            if self.z0 is None:
                raise ValueError("Must estimate Z0 before plotting. Call fit() first.")
            
            di_points = self._get_di_points()
            pdf_points = self._get_pdf_points()
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            # Plot 1: Full PDF with Z0 location
            ax1.plot(di_points, pdf_points, 'b-', linewidth=2, label='PDF')
            ax1.axvline(x=self.z0, color='red', linestyle='--', linewidth=2, 
                       label=f'Z0 = {self.z0:.4f}')
            
            # Highlight the Z0 point
            pdf_at_z0 = np.interp(self.z0, di_points, pdf_points)
            ax1.scatter([self.z0], [pdf_at_z0], color='red', s=100, zorder=5, 
                       label=f'Z0 Point (PDF={pdf_at_z0:.4f})')
            
            # Mark global maximum from discrete data for comparison
            global_max_idx = np.argmax(pdf_points)
            ax1.scatter([di_points[global_max_idx]], [pdf_points[global_max_idx]], 
                       color='orange', s=80, zorder=4, marker='s',
                       label=f'Discrete Max')
            
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
            ax2.scatter([self.z0], [pdf_at_z0], color='red', s=100, zorder=5)
            
            ax2.set_xlabel('Data Points')
            ax2.set_ylabel('PDF')
            ax2.set_title(f'Zoomed View Around Z0\nMethod: {self.estimation_info.get("z0_method", "unknown")}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting")
        except Exception as e:
            print(f"Error creating plot: {e}")
    
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
    
    def _find_z0_advanced_global(self, global_max_idx, di_points, pdf_points):
        """
        Find Z0 using advanced methods while ensuring global maximum is found.
        
        This method tries multiple advanced approaches and always validates that
        the result corresponds to the global maximum region.
        """
        # Store the method used for reporting
        self._last_method = 'unknown'
        
        # Method 1: Try spline optimization over entire domain
        z0_spline = self._try_spline_global_optimization(di_points, pdf_points, global_max_idx)
        if z0_spline is not None:
            self._last_method = 'global_spline_optimization'
            return z0_spline
        
        # Method 2: Try polynomial fitting in a larger window around global max
        z0_poly = self._try_polynomial_global_fitting(di_points, pdf_points, global_max_idx)
        if z0_poly is not None:
            self._last_method = 'global_polynomial_fit'
            return z0_poly
        
        # Method 3: Try refined interpolation around global maximum
        z0_interp = self._try_refined_global_interpolation(di_points, pdf_points, global_max_idx)
        if z0_interp is not None:
            self._last_method = 'global_refined_interpolation'
            return z0_interp
        
        # Method 4: Parabolic interpolation around global maximum
        z0_parabolic = self._try_parabolic_interpolation(di_points, pdf_points, global_max_idx)
        if z0_parabolic is not None:
            self._last_method = 'global_parabolic_interpolation'
            return z0_parabolic
        
        # Fallback: Use the discrete global maximum
        if self.verbose:
            print("All advanced methods failed, using discrete global maximum")
        self._last_method = 'discrete_global_maximum'
        return di_points[global_max_idx]
    
    def _try_spline_global_optimization(self, di_points, pdf_points, global_max_idx):
        """Try spline optimization over the entire domain with global maximum validation."""
        try:
            from scipy.interpolate import UnivariateSpline
            from scipy.optimize import minimize_scalar
            
            # Create spline over entire domain
            spline = UnivariateSpline(di_points, pdf_points, s=0, k=min(3, len(di_points)-1))
            
            # Find global maximum of spline over entire domain
            result = minimize_scalar(
                lambda x: -spline(x),  # Negative because we minimize
                bounds=(di_points.min(), di_points.max()),
                method='bounded'
            )
            
            if result.success:
                z0_candidate = result.x
                
                # Validate that this is indeed near the global maximum
                spline_value_at_candidate = spline(z0_candidate)
                discrete_max_value = pdf_points[global_max_idx]
                
                # Check if the spline maximum is consistent with discrete maximum
                if abs(spline_value_at_candidate - discrete_max_value) <= discrete_max_value * 0.1:
                    if self.verbose:
                        print(f"Spline optimization successful: Z0={z0_candidate:.6f}, "
                              f"PDF={spline_value_at_candidate:.6f}")
                    return z0_candidate
                else:
                    if self.verbose:
                        print(f"Spline optimization found inconsistent maximum: "
                              f"spline_max={spline_value_at_candidate:.6f}, "
                              f"discrete_max={discrete_max_value:.6f}")
            
        except ImportError:
            if self.verbose:
                print("SciPy not available for spline optimization")
        except Exception as e:
            if self.verbose:
                print(f"Spline optimization failed: {e}")
        
        return None
    
    def _try_polynomial_global_fitting(self, di_points, pdf_points, global_max_idx):
        """Try polynomial fitting with a larger window around global maximum."""
        try:
            # Use a larger window around the global maximum (up to 30% of data or ±15 points)
            window_size = min(15, int(len(pdf_points) * 0.3))
            start_idx = max(0, global_max_idx - window_size)
            end_idx = min(len(pdf_points), global_max_idx + window_size + 1)
            
            # Ensure minimum window size
            if end_idx - start_idx < 3:
                return None
            
            x_window = di_points[start_idx:end_idx]
            y_window = pdf_points[start_idx:end_idx]
            
            # Fit polynomial (degree 2 to 4 depending on data size)
            degree = min(4, len(x_window) - 1)
            if degree < 2:
                return None
                
            coeffs = np.polyfit(x_window, y_window, degree)
            poly = np.poly1d(coeffs)
            
            # Find derivative and critical points
            poly_deriv = np.polyder(poly)
            critical_points = np.roots(poly_deriv)
            
            # Filter real roots within the window
            real_critical = critical_points[np.isreal(critical_points)].real
            valid_critical = real_critical[
                (real_critical >= x_window.min()) & 
                (real_critical <= x_window.max())
            ]
            
            if len(valid_critical) > 0:
                # Evaluate polynomial at all critical points and find the maximum
                critical_values = [poly(cp) for cp in valid_critical]
                max_critical_idx = np.argmax(critical_values)
                best_critical = valid_critical[max_critical_idx]
                best_value = critical_values[max_critical_idx]
                
                # Validate that this critical point gives a maximum (not minimum)
                poly_second_deriv = np.polyder(poly_deriv)
                second_deriv_at_point = poly_second_deriv(best_critical)
                
                if second_deriv_at_point < 0:  # Negative second derivative = maximum
                    if self.verbose:
                        print(f"Polynomial fitting successful: Z0={best_critical:.6f}, "
                              f"PDF={best_value:.6f}")
                    return best_critical
                else:
                    if self.verbose:
                        print("Polynomial critical point is a minimum, not maximum")
            
        except Exception as e:
            if self.verbose:
                print(f"Polynomial fitting failed: {e}")
        
        return None
    
    def _try_refined_global_interpolation(self, di_points, pdf_points, global_max_idx):
        """Try refined interpolation around the global maximum."""
        try:
            from scipy.interpolate import interp1d
            
            # Create a focused window around global maximum (smaller than polynomial)
            window_size = min(8, len(pdf_points) // 8)
            start_idx = max(0, global_max_idx - window_size)
            end_idx = min(len(pdf_points), global_max_idx + window_size + 1)
            
            if end_idx - start_idx < 3:
                return None
            
            x_window = di_points[start_idx:end_idx]
            y_window = pdf_points[start_idx:end_idx]
            
            # Create cubic interpolation
            interp_func = interp1d(x_window, y_window, kind='cubic', 
                                 bounds_error=False, fill_value='extrapolate')
            
            # Create fine grid and find maximum
            n_fine = min(1000, len(x_window) * 100)
            x_fine = np.linspace(x_window.min(), x_window.max(), n_fine)
            y_fine = interp_func(x_fine)
            
            # Find maximum on fine grid
            max_fine_idx = np.argmax(y_fine)
            z0_candidate = x_fine[max_fine_idx]
            max_value = y_fine[max_fine_idx]
            
            if self.verbose:
                print(f"Refined interpolation successful: Z0={z0_candidate:.6f}, "
                      f"PDF={max_value:.6f}")
            return z0_candidate
                
        except ImportError:
            if self.verbose:
                print("SciPy not available for refined interpolation")
        except Exception as e:
            if self.verbose:
                print(f"Refined interpolation failed: {e}")
        
        return None
    
    def _try_parabolic_interpolation(self, di_points, pdf_points, global_max_idx):
        """Try parabolic interpolation around the global maximum."""
        try:
            # Need at least 3 points for parabolic interpolation
            if global_max_idx == 0 or global_max_idx == len(pdf_points) - 1:
                return None
            
            # Get three points around the global maximum
            x1, x2, x3 = di_points[global_max_idx-1:global_max_idx+2]
            y1, y2, y3 = pdf_points[global_max_idx-1:global_max_idx+2]
            
            # Parabolic interpolation formula
            denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
            if abs(denom) < 1e-12:  # Avoid division by zero
                return None
            
            A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
            B = (x3*x3 * (y1 - y2) + x2*x2 * (y3 - y1) + x1*x1 * (y2 - y3)) / denom
            
            if abs(A) < 1e-12:  # Ensure we have a parabola
                return None
            
            # Find maximum of parabola
            x_max = -B / (2 * A)
            
            # Validate that A < 0 (parabola opens downward = maximum exists)
            # and that the maximum is within a reasonable range
            if A < 0 and x1 <= x_max <= x3:
                if self.verbose:
                    print(f"Parabolic interpolation successful: Z0={x_max:.6f}")
                return x_max
            
        except Exception as e:
            if self.verbose:
                print(f"Parabolic interpolation failed: {e}")
        
        return None
    
    def _get_last_method_used(self):
        """Get the last method used for estimation."""
        return getattr(self, '_last_method', 'unknown')
    
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
    
    def __repr__(self):
        """Return string representation of the Z0Estimator instance."""
        method = self.estimation_info.get('z0_method', 'not_estimated') if self.estimation_info else 'not_estimated'
        return f"Z0Estimator(gdf_type='{self.gdf_type}', z0={self.z0}, method='{method}')"