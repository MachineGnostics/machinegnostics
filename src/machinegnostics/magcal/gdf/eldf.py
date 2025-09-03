'''
ELDF

Machine Gnostics
Author: Nirmal Parmar
'''

import numpy as np
from machinegnostics.magcal.gdf.base_eldf import BaseELDF

class ELDF(BaseELDF):
    """
    ELDF - Estimating Local Distribution Function.

    A comprehensive class for estimating and analyzing gnostic local distribution functions for given data.
    This class provides methods to fit local distribution functions and visualize results with
    optional bounds, weighting capabilities, and advanced Z0 (Gnostic Mean) point estimation for maximum PDF location.

    The ELDF class supports both additive and multiplicative data forms and can handle bounded and
    unbounded data distributions. It provides automatic parameter estimation with flexible 
    visualization options and advanced interpolation methods for precise local distribution analysis.

    The Estimating Local Distribution Function (ELDF) is a gnostic model that estimates 
    the underlying local distribution characteristics of data points while accounting for various 
    constraints and bounds. It uses advanced optimization techniques including iterative Z0 estimation 
    to find the best-fitting parameters and can handle weighted data for improved accuracy in specific 
    applications. Unlike global distribution functions, ELDF focuses on local characteristics and 
    provides detailed PDF analysis around critical points.

    Key Features:
        - Automatic parameter estimation with customizable bounds
        - Advanced Z0 point estimation using multiple interpolation methods
        - Support for weighted data points with WEDF integration
        - Multiple data processing forms (additive/multiplicative)
        - Comprehensive visualization capabilities with PDF and ELDF plots
        - Robust optimization with multiple solver options
        - Memory-efficient processing for large datasets
        - Iterative optimization for Z0 convergence with early stopping
        - Spline, polynomial, and refined interpolation methods
        - Built-in homogeneity analysis and data quality assessment

    Attributes:
        data (np.ndarray): The input dataset used for local distribution estimation.
        DLB (float): Data Lower Bound - absolute minimum value the data can take.
        DUB (float): Data Upper Bound - absolute maximum value the data can take.
        LB (float): Lower Probable Bound - practical lower limit for the distribution.
        UB (float): Upper Probable Bound - practical upper limit for the distribution.
        S (float or str): Scale parameter for the distribution. Set to 'auto' for automatic estimation. Consider using S value in range [0, 2].
        z0_optimize (bool): Whether to use advanced optimization for Z0 point estimation (default: True).
        tolerance (float): Convergence tolerance for optimization (default: 1e-5).
        data_form (str): Form of the data processing:
            - 'a': Additive form (default) - treats data linearly
            - 'm': Multiplicative form - applies logarithmic transformation
        n_points (int): Number of points to generate in the distribution function (default: 1000).
        homogeneous (bool): Whether to assume data homogeneity for optimization strategies (default: True).
        catch (bool): Whether to store intermediate calculated values (default: True).
        weights (np.ndarray): Prior weights for data points. If None, uniform weights are used.
        wedf (bool): Whether to use Weighted Empirical Distribution Function (default: True).
        opt_method (str): Optimization method for parameter estimation (default: 'L-BFGS-B').
        verbose (bool): Whether to print detailed progress information (default: False).
        max_data_size (int): Maximum data size for smooth ELDF generation (default: 1000).
        flush (bool): Whether to flush large arrays during processing (default: True).
        z0 (float): The Z0 point where PDF reaches its maximum (computed after fitting).
        params (dict): Dictionary storing fitted parameters and results after fitting.

    Methods:
        fit(plot=False): Fit the Estimating Local Distribution Function to the data.
        plot(plot_smooth=True, plot='both', bounds=True, extra_df=True, figsize=(12,8)): 
            Visualize the fitted local distribution with PDF and ELDF curves.
    
    Examples:
        Basic usage with default parameters:
        >>> import numpy as np
        >>> from machinegnostics.magcal.eldf import ELDF
        >>> 
        >>> # Stack Loss example data
        >>> data = [7, 8, 8, 8, 9, 11, 12, 13, 14, 14, 15, 15, 15, 18, 18, 19, 20, 28, 37, 37, 42]
        >>> data = np.array(data)
        >>> eldf = ELDF(data)
        >>> eldf.fit()
        >>> eldf.plot()
        >>> 
        >>> # Access Z0 point and fitted parameters
        >>> print(f"Z0 point (max PDF): {eldf.z0:.6f}")
        >>> print(f"Fitted parameters: {eldf.params}")
        
        Usage with custom bounds, weights, and Z0 optimization:
        >>> 
        >>> # Fit with bounds and advanced Z0 estimation
        >>> weights = np.random.uniform(0.8, 1.2, len(data))
        >>> eldf = ELDF(data, LB=5, UB=50, weights=weights, z0_optimize=True, verbose=True)
        >>> eldf.fit()
        >>> eldf.plot(bounds=True)
        >>> print(f"Z0 estimation method: {eldf.params.get('z0_method', 'not_available')}")
        
        High-resolution analysis with custom tolerance:
        >>> # For precise local distribution analysis
        >>> data = np.random.gamma(2, 2, 100)
        >>> eldf = ELDF(data, n_points=2000, tolerance=1e-6, z0_optimize=True)
        >>> eldf.fit()
        >>> eldf.plot(plot='both')  # Show both PDF and ELDF
    
    Workflow:
        1. Initialize ELDF with your data and desired parameters
        2. Call fit() to estimate the local distribution parameters and Z0 point
        3. Use plot() to visualize the results including PDF analysis
        4. Access Z0 point and other parameters for further analysis
        
        >>> eldf = ELDF(data, DLB=0, UB=100, z0_optimize=True)  # Step 1: Initialize
        >>> eldf.fit()                                          # Step 2: Fit
        >>> eldf.plot(bounds=True, plot='both')                 # Step 3: Visualize
        >>> z0_point = eldf.z0                                  # Step 4: Extract Z0
    
    Z0 Point Estimation:
        The Z0 point represents the location where the PDF reaches its maximum. ELDF provides
        multiple estimation methods:
        - Spline optimization: Uses scipy splines for smooth interpolation
        - Polynomial fitting: Fits polynomials around the maximum for analytical solutions
        - Refined interpolation: Creates fine grids using cubic interpolation
        - Simple maximum: Direct maximum finding with parabolic interpolation
        
        The method is automatically selected based on data characteristics and smoothness.
    
    Performance Tips:
        - Use data_form='m' for multiplicative/log-normal data
        - Set appropriate bounds to improve convergence and Z0 accuracy
        - Use z0_optimize=True for precise Z0 estimation, False for speed
        - Use catch=False for large datasets to save memory
        - Adjust n_points based on visualization needs vs. performance
        - Use verbose=True to monitor Z0 optimization progress
        - For repeated analysis, save fitted parameters and Z0 values
        - Increase tolerance for faster convergence, decrease for precision
    
    Common Use Cases:
        - Peak detection and modal analysis in data distributions
        - Local density estimation for clustering applications
        - Risk analysis focusing on critical value identification
        - Quality control with emphasis on specification limits
        - Financial modeling with focus on maximum likelihood points
        - Environmental monitoring around critical thresholds
        - Biostatistics for identifying characteristic response levels
        - Engineering analysis for optimal operating points
        - Process optimization around performance peaks
        - Anomaly detection based on local distribution characteristics
    
    Notes:
        - Bounds (DLB, DUB, LB, UB) are optional but improve Z0 estimation accuracy
        - When S='auto', the scale parameter is automatically estimated from the data
        - The weights array must have the same length as the data array
        - Setting catch=False saves memory but prevents access to detailed results
        - Z0 optimization uses iterative methods that may take additional time
        - Different interpolation methods are automatically selected based on data quality
        - The verbose flag provides detailed output for both fitting and Z0 estimation
        - PDF analysis is particularly useful for understanding local distribution behavior
        - ELDF provides more detailed local analysis compared to global distribution functions
        
    Raises:
        ValueError: If data array is empty or contains invalid values.
        ValueError: If weights array length doesn't match data array length.
        ValueError: If bounds are specified incorrectly (e.g., LB > UB).
        ValueError: If invalid parameters are provided (negative tolerance, invalid data_form, etc.).
        RuntimeError: If the fitting process fails to converge.
        OptimizationError: If the optimization algorithm encounters numerical issues.
        ConvergenceError: If Z0 optimization fails to converge within specified tolerance.
        ImportError: If required dependencies (matplotlib for plotting, scipy for advanced Z0 methods) are not available.
        
    Advanced Features:
        Z0 Convergence Monitoring:
        - Early stopping with patience parameter
        - Multiple fallback methods if primary optimization fails
        - Gradient clipping and momentum for stable convergence
        - Adaptive learning rates with cooling schedules
        
        Interpolation Method Selection:
        - Automatic method selection based on data smoothness
        - Spline optimization for smooth, high-resolution data
        - Polynomial fitting for moderate-resolution data
        - Refined interpolation for general cases
        - Simple maximum with parabolic interpolation as fallback
        
        Memory Management:
        - Automatic flushing of large intermediate arrays
        - Configurable maximum data size limits
        - Optional catching of intermediate results
        - Efficient storage of essential parameters only
        
    Troubleshooting:
        Z0 Estimation Issues:
        - If Z0 optimization doesn't converge, try increasing tolerance
        - For oscillating behavior, the algorithm uses adaptive learning rates
        - Check data quality and consider preprocessing if needed
        - Use verbose=True to monitor convergence progress
        
        Fitting Issues:
        - Verify bounds are reasonable and not too restrictive
        - Check for NaN or infinite values in data
        - Try different optimization methods for difficult datasets
        - Increase n_points for better resolution if needed
        
        Performance Issues:
        - Set z0_optimize=False for faster fitting if Z0 is not critical
        - Reduce n_points for faster computation
        - Use catch=False for memory-constrained environments
        - Consider data sampling for extremely large datasets
        
    """

    def __init__(self,
                 data: np.ndarray,
                 DLB: float = None,
                 DUB: float = None,
                 LB: float = None,
                 UB: float = None,
                 S = 'auto',
                 varS: bool = False,
                 z0_optimize: bool = True,
                 tolerance: float = 1e-5,
                 data_form: str = 'a',
                 n_points: int = 1000,
                 homogeneous: bool = True,
                 catch: bool = True,
                 weights: np.ndarray = None,
                 wedf: bool = True,
                 opt_method: str = 'L-BFGS-B',
                 verbose: bool = False,
                 max_data_size: int = 1000,
                 flush: bool = True):
        super().__init__(data=data,
                         DLB=DLB,
                         DUB=DUB,
                         LB=LB,
                         UB=UB,
                         S=S,
                         varS=varS,
                         z0_optimize=z0_optimize,
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

    def fit(self, plot: bool = False):
        """
        Fit the Estimating Local Distribution Function (ELDF) model to the data.
        
        This method performs the complete ELDF fitting process including parameter optimization,
        Z0 point estimation, PDF calculation, and optional visualization. The fitting process
        automatically estimates the best parameters for the local distribution function and
        computes the Z0 point where the PDF reaches its maximum using advanced interpolation
        methods when z0_optimize=True.
        
        The fitting process includes:
        1. Data preprocessing and transformation based on data_form
        2. Automatic bound estimation if not provided
        3. Scale parameter optimization if S='auto'
        4. ELDF core computation with iterative optimization
        5. PDF calculation for local density analysis
        6. Z0 point estimation using interpolation methods
        7. Optional smooth curve generation for visualization
        8. Parameter storage and validation
        
        Parameters:
        -----------
        plot : bool, optional (default=False)
            Whether to automatically display a plot of the fitted ELDF results after fitting.
            If True, calls the plot() method with default parameters to show both ELDF and PDF
            curves with bounds and additional distribution functions.
            
            - True: Display plot immediately after fitting
            - False: Fit silently without plotting (plot can be called separately later)
        
        Returns:
        --------
        None
            The method modifies the ELDF object in-place, storing all fitted parameters
            and results in the following attributes:
            
            - self.z0: The Z0 point where PDF is maximum
            - self.params: Dictionary containing all fitted parameters and metadata
            - self.eldf_points: Smooth ELDF curve points (if n_points > 0)
            - self.pdf_points: Smooth PDF curve points (if n_points > 0)
        
        Raises:
        -------
        ValueError
            - If data array is empty or contains invalid values (NaN, inf)
            - If weights array length doesn't match data array length
            - If bounds are specified incorrectly (e.g., LB > UB, DLB > DUB)
            - If invalid parameters are provided (negative tolerance, invalid data_form)
            - If data_form is not 'a' (additive) or 'm' (multiplicative)
            - If n_points is negative or zero when smooth curves are requested
            
        RuntimeError
            - If the optimization algorithm fails to converge
            - If Z0 estimation fails with all available methods
            - If numerical instabilities prevent successful fitting
            
        OptimizationError
            - If the chosen optimization method encounters numerical issues
            - If parameter bounds are too restrictive for convergence
            
        ConvergenceError
            - If Z0 optimization fails to converge within specified tolerance
            - If iterative optimization oscillates without improvement
            
        ImportError
            - If required dependencies are not available for specific features
            - If scipy is needed for advanced Z0 estimation but not installed
        
        Notes:
        ------
        - The fitting process automatically selects the best Z0 estimation method based on
        data characteristics and smoothness when z0_optimize=True
        - For large datasets (> max_data_size), consider setting plot_smooth=False in
        subsequent plot() calls for better performance
        - The catch parameter must be True to access fitted parameters and plotting
        - Verbose output can be enabled during initialization to monitor fitting progress
        - The tolerance parameter affects both general optimization and Z0 estimation precision
        - Different optimization methods can be specified via opt_method parameter
        
        Performance Considerations:
        ---------------------------
        - Fitting time increases with data size and n_points
        - Z0 optimization adds computational overhead but provides precise maximum location
        - Setting z0_optimize=False speeds up fitting if Z0 is not critical
        - Memory usage scales with n_points and data size
        - Use flush=True for memory-constrained environments
        
        Post-Fitting Analysis:
        ----------------------
        After successful fitting, the following analyses become available:
        - Z0 point analysis for maximum PDF location
        - Local distribution characteristics via ELDF curves
        - Probability density analysis via PDF curves
        - Statistical parameter interpretation via params dictionary
        - Visualization via plot() method with various options
        
        Examples:
        ---------
        Basic fitting with default parameters:
        >>> eldf = ELDF(data)
        >>> eldf.fit()
        >>> print(f"Z0 point: {eldf.z0:.6f}")
        
        Fitting with immediate visualization:
        >>> eldf = ELDF(data, verbose=True)
        >>> eldf.fit(plot=True)  # Fits and plots immediately
        
        High-precision fitting with custom tolerance:
        >>> eldf = ELDF(data, tolerance=1e-6, z0_optimize=True)
        >>> eldf.fit()
        >>> print(f"Z0 method: {eldf.params['z0_method']}")
        
        Memory-efficient fitting for large datasets:
        >>> eldf = ELDF(large_data, catch=False, z0_optimize=False)
        >>> eldf.fit()  # Fast fitting without storing intermediate results
        
        """
        self._fit_eldf(plot=plot)
    
    def plot(self, plot_smooth=True,
                   plot='both',
                   bounds=True,
                   extra_df=True,
                   figsize=(12, 8)):
        """
        Visualize the fitted Estimating Local Distribution Function (ELDF) results.
        
        This method creates comprehensive visualizations of the fitted ELDF model, including
        the local distribution function, probability density function (PDF), bounds, and
        optional additional distribution functions. The plot includes the Z0 point marking
        where the PDF reaches its maximum, providing insight into the local distribution
        characteristics and critical values.
        
        The visualization automatically adapts based on available data and chosen options,
        providing both discrete and smooth curve representations with professional styling
        and comprehensive legends.
        
        Parameters:
        -----------
        plot_smooth : bool, optional (default=True)
            Whether to include smooth interpolated curves in the plot.
            
            - True: Display smooth curves using n_points interpolation for ELDF and PDF
            - False: Display only discrete points from the original data
            
            Note: Smooth plotting requires successful fitting with n_points > 0.
            For large datasets (> max_data_size), a warning is displayed recommending
            plot_smooth=False for better performance.
            
        plot : str, optional (default='both')
            Specifies which components to include in the visualization.
            
            - 'gdf' or 'eldf': Display only the Estimating Local Distribution Function
            - 'pdf': Display only the Probability Density Function
            - 'both': Display both ELDF and PDF (PDF on secondary y-axis)
            
            When 'both' is selected, ELDF is plotted on the primary y-axis (left) and
            PDF on the secondary y-axis (right) for easy comparison.
            
        bounds : bool, optional (default=True)
            Whether to display bound lines and regions on the plot.
            
            - True: Show DLB, DUB, LB, UB bounds as vertical lines with labels
            - False: Hide bound indicators for cleaner visualization
            
            Bound lines help identify the data range and probable bounds used in fitting.
            
        extra_df : bool, optional (default=True)
            Whether to include additional distribution functions in the plot.
            
            - True: Display WEDF (Weighted Empirical Distribution Function) and 
                    KSDF (Kernel Smoothed Distribution Function) if available
            - False: Show only the main ELDF and PDF curves
            
            Extra distribution functions provide comparative analysis with empirical
            and kernel-smoothed estimates.
            
        figsize : tuple, optional (default=(12, 8))
            Figure size specification as (width, height) in inches.
            Larger sizes provide better detail for complex plots with multiple curves.
            
            Example sizes:
            - (10, 6): Compact size for simple plots
            - (12, 8): Default balanced size
            - (15, 10): Large size for detailed analysis
            - (8, 6): Smaller size for embedded plots
        
        Returns:
        --------
        None
            The method displays the plot using matplotlib and does not return any value.
            The plot is shown interactively and can be saved manually or programmatically
            using matplotlib's savefig() function after calling this method.
        
        Raises:
        -------
        RuntimeError
            - If the ELDF model has not been fitted before plotting
            - If required plot data is not available due to fitting issues
            
        ValueError
            - If the plot parameter is not one of: 'gdf', 'eldf', 'pdf', 'both'
            - If figsize is not a valid tuple of positive numbers
            - If requested plot components are not available (e.g., PDF not computed)
            
        ImportError
            - If matplotlib is not available for plotting
            - If required plotting dependencies are missing
        
        Warnings:
        ---------
        Performance Warning
            Displayed when plot_smooth=True and data size exceeds max_data_size,
            suggesting to use plot_smooth=False for better performance.
            
        Data Availability Warning
            Shown when catch=False was used during fitting, limiting plot capabilities.
        
        Plot Components:
        ----------------
        The plot may include the following elements based on parameters:
        
        Primary Components:
        - ELDF curve: Main local distribution function (blue line)
        - PDF curve: Probability density function (red line/points)
        - Z0 line: Vertical magenta dash-dot line marking maximum PDF location
        - Data points: Original data scatter points
        
        Bound Indicators (if bounds=True):
        - DLB: Data Lower Bound (green solid line)
        - DUB: Data Upper Bound (brown solid line)
        - LB: Lower Bound (purple dashed line)
        - UB: Upper Bound (purple dashed line)
        
        Additional Functions (if extra_df=True):
        - WEDF: Weighted Empirical Distribution Function (gray dots)
        - KSDF: Kernel Smoothed Distribution Function (cyan line)
        
        Legends and Labels:
        - Comprehensive legend identifying all plotted elements
        - Axis labels with appropriate units and descriptions
        - Title indicating ELDF analysis with key parameters
        - Grid for easier value reading
        
        Examples:
        ---------
        Basic plotting after fitting:
        >>> eldf = ELDF(data)
        >>> eldf.fit()
        >>> eldf.plot()  # Shows both ELDF and PDF with all features
        
        PDF-only visualization:
        >>> eldf.plot(plot='pdf', bounds=False)  # Clean PDF plot
        
        Performance-optimized plotting for large datasets:
        >>> eldf.plot(plot_smooth=False, extra_df=False, figsize=(10, 6))
        
        High-detail analysis plot:
        >>> eldf.plot(plot_smooth=True, bounds=True, extra_df=True, figsize=(15, 10))
        
        Minimal ELDF visualization:
        >>> eldf.plot(plot='eldf', bounds=False, extra_df=False)
        
        Custom size for presentation:
        >>> eldf.plot(figsize=(16, 9))  # Wide format for presentations
        
        Interactive Analysis Workflow:
        ------------------------------
        >>> # Fit and explore different visualizations
        >>> eldf = ELDF(data, verbose=True)
        >>> eldf.fit()
        >>> 
        >>> # Overview with all components
        >>> eldf.plot(plot='both', bounds=True, extra_df=True)
        >>> 
        >>> # Focus on PDF analysis around Z0
        >>> eldf.plot(plot='pdf', bounds=True)
        >>> print(f"Z0 point: {eldf.z0:.6f}")
        >>> 
        >>> # Clean ELDF visualization
        >>> eldf.plot(plot='eldf', bounds=False, extra_df=False)
        
        Performance Tips:
        -----------------
        - Use plot_smooth=False for datasets larger than max_data_size
        - Set extra_df=False to reduce plot complexity and rendering time
        - Choose appropriate figsize based on display resolution and detail needs
        - For batch processing, consider saving plots programmatically:
        
        >>> eldf.plot()
        >>> plt.savefig('eldf_analysis.png', dpi=300, bbox_inches='tight')
        >>> plt.close()
        
        Interpretation Guide:
        ---------------------
        - Z0 (Gnostic Mean) line indicates the most probable value (PDF maximum)
        - ELDF curve shows cumulative local distribution characteristics
        - PDF curve shows local density and concentration of data
        - Bound lines indicate the effective range of the distribution
        - Steep ELDF regions correspond to high PDF values
        - Flat ELDF regions correspond to low probability areas
        
        Troubleshooting:
        ----------------
        Plot Not Displaying:
        - Ensure matplotlib backend is properly configured
        - Check if running in headless environment (use savefig instead)
        - Verify that fit() was called successfully before plotting
        
        Missing Plot Elements:
        - If Z0 line is missing, check that z0_optimize=True during fitting
        - If smooth curves are missing, ensure n_points > 0 during initialization
        - If bounds are missing, verify that bounds were set during initialization
        
        Performance Issues:
        - Reduce n_points for faster smooth curve generation
        - Use plot_smooth=False for immediate rendering
        - Consider smaller figsize for faster display
        """
        self._plot(plot_smooth=plot_smooth,
                   plot=plot,
                   bounds=bounds,
                   extra_df=extra_df,
                   figsize=figsize)