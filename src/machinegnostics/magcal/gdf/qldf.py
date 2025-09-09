'''
QLDF Quantifying Local Distribution Function (QLDF)

Public class
Author: Nirmal Parmar
Machine Gnostics
'''

import numpy as np
from machinegnostics.magcal.gdf.base_qldf import BaseQLDF

class QLDF(BaseQLDF):
    """
    QLDF - Quantifying Local Distribution Function.
    
    A comprehensive class for quantifying and analyzing local distribution characteristics around critical points (inliers) in given data.
    This class provides methods to fit local distribution functions with focus on identifying and analyzing points of minimum 
    probability density (Z0 points) and their local neighborhood behavior.

    The QLDF class specializes in local distribution analysis, making it particularly valuable for identifying critical points,
    local minima in probability density, and understanding localized distribution behavior. Unlike global distribution methods,
    QLDF focuses on detailed characterization of specific regions within the data distribution.

    The Quantifying Local Distribution Function (QLDF) is a gnostic-probabilistic model that identifies and quantifies 
    local distribution characteristics, particularly focusing on regions of low probability density. It uses advanced 
    optimization techniques to locate Z0 points (global PDF minima) and provides detailed analysis of local distribution 
    behavior around these critical points.

    Key Features:
        - Automatic Z0 point identification (global PDF minimum)
        - Local distribution characterization around critical points
        - Advanced interpolation methods for precise Z0 estimation
        - Support for weighted data analysis
        - Memory-efficient processing for large datasets
        - Comprehensive visualization of local distribution features
        - Robust optimization with multiple solver options
        - Integration with empirical distribution methods

    Attributes:
        data (np.ndarray): The input dataset used for local distribution analysis.
        DLB (float): Data Lower Bound - absolute minimum value the data can take.
        DUB (float): Data Upper Bound - absolute maximum value the data can take.
        LB (float): Lower Probable Bound - practical lower limit for the distribution.
        UB (float): Upper Probable Bound - practical upper limit for the distribution.
        S (float or str): Scale parameter for the distribution. Set to 'auto' for automatic estimation.
        varS (bool): Whether to use variable scale parameter during optimization (default: False).
        z0_optimize (bool): Whether to optimize the location parameter z0 during fitting (default: True).
        data_form (str): Form of the data processing:
            - 'a': Additive form (default) - treats data linearly
            - 'm': Multiplicative form - applies logarithmic transformation
        n_points (int): Number of points to generate in the distribution function (default: 500).
        catch (bool): Whether to store intermediate calculated values (default: True).
        weights (np.ndarray): Prior weights for data points. If None, uniform weights are used.
        wedf (bool): Whether to use Weighted Empirical Distribution Function (default: False for better performance).
        opt_method (str): Optimization method for parameter estimation (default: 'L-BFGS-B').
        tolerance (float): Convergence tolerance for optimization (default: 1e-5).
        verbose (bool): Whether to print detailed progress information (default: False).
        params (dict): Dictionary storing fitted parameters and results after fitting.
        homogeneous (bool): To indicate data homogeneity (default: True).
        max_data_size (int): Maximum data size for smooth QLDF generation (default: 1000).
        flush (bool): Whether to flush large arrays (default: True).
        z0 (float): Identified Z0 point (global PDF minimum) after fitting.

    Methods:
        fit(): Fit the Quantifying Local Distribution Function to the data.
        plot(plot_smooth=True, plot='both', bounds=True, extra_df=True, figsize=(12,8)): 
            Visualize the fitted local distribution with Z0 point identification.
        results(): Get comprehensive results including Z0 location and local distribution parameters.
    
    Examples:
        Basic usage with automatic Z0 identification:
        >>> import numpy as np
        >>> from machinegnostics.magcal import QLDF
        >>> 
        >>> # Bimodal data with clear minimum between modes
        >>> data = np.concatenate([np.random.normal(-2, 0.5, 50), 
        ...                       np.random.normal(2, 0.5, 50)])
        >>> qldf = QLDF(data)
        >>> qldf.fit()
        >>> print(f"Z0 point (PDF minimum): {qldf.z0:.4f}")
        >>> qldf.plot()
        
        Usage with custom bounds and optimization:
        >>> # Financial return data analysis
        >>> returns = np.array([-0.05, -0.02, 0.01, 0.03, -0.01, 0.02, 0.04, -0.03])
        >>> qldf = QLDF(returns, LB=-0.1, UB=0.1, z0_optimize=True)
        >>> qldf.fit()
        >>> results = qldf.results()
        >>> print(f"Critical point at: {results['z0']:.6f}")
        >>> qldf.plot(bounds=True)
        
        Advanced Z0 analysis with variable scale:
        >>> # Quality control data with specification limits
        >>> measurements = np.random.beta(2, 5, 100) * 10  # Skewed data
        >>> qldf = QLDF(measurements, 
        ...              DLB=0, DUB=10,
        ...              varS=True,
        ...              tolerance=1e-6,
        ...              verbose=True)
        >>> qldf.fit()
        >>> qldf.plot(plot='both', bounds=True)
        
        Memory-efficient processing for large datasets:
        >>> # Large dataset analysis
        >>> large_data = np.concatenate([
        ...     np.random.exponential(2, 10000),
        ...     np.random.exponential(5, 10000)
        ... ])
        >>> qldf = QLDF(large_data,
        ...              catch=False,  # Save memory
        ...              n_points=200,
        ...              max_data_size=15000)
        >>> qldf.fit()
        >>> print(f"Z0 at PDF minimum: {qldf.z0:.4f}")
    
    Workflow:
        1. Initialize QLDF with your data and desired parameters
        2. Call fit() to identify Z0 point and estimate local distribution parameters
        3. Use plot() to visualize Z0 location and local distribution characteristics
        4. Access detailed results for further analysis
        
        >>> qldf = QLDF(data, DLB=0, UB=100)  # Step 1: Initialize
        >>> qldf.fit()                        # Step 2: Fit and find Z0
        >>> qldf.plot(plot='both')           # Step 3: Visualize
        >>> results = qldf.results()         # Step 4: Get detailed results
    
    Performance Tips:
        - Set wedf=False for better performance (default recommendation)
        - Use varS=True only when scale parameter variation is expected
        - Set appropriate bounds to improve Z0 identification accuracy
        - Use catch=False for large datasets to save memory
        - Adjust tolerance based on required Z0 precision
        - Use verbose=True to monitor Z0 optimization progress
        - For repeated analysis, save fitted parameters and Z0 location
    
    Common Use Cases:
        - Failure analysis and reliability engineering (identifying critical stress points)
        - Quality control (finding specification limit intersections)
        - Financial risk analysis (identifying market stress points)
        - Process optimization (locating optimal operating points)
        - Biostatistics (identifying threshold effects)
        - Environmental monitoring (detecting transition points)
        - Signal processing (identifying signal discontinuities)
        - Market analysis (finding price support/resistance levels)
    
    Z0 Point Applications:
        The Z0 point (global PDF minimum) identified by QLDF is particularly valuable for:
        - Risk management: Points of maximum uncertainty or transition
        - Quality control: Specification boundaries and acceptable limits
        - Process control: Operating points with minimum process variation
        - Reliability engineering: Stress levels with minimum failure probability density
        - Financial modeling: Market conditions with minimum activity
        - Scientific analysis: Phase transitions and critical phenomena
    
    Notes:
        - Z0 optimization is enabled by default and highly recommended for accurate results
        - The wedf parameter is set to False by default as it works better without KSDF for local analysis
        - Bounds (DLB, DUB, LB, UB) significantly influence Z0 identification accuracy
        - Variable scale (varS=True) should be used when local distribution characteristics vary significantly
        - The Z0 point represents the global minimum of the PDF, not a local minimum
        - QLDF focuses on local distribution characteristics rather than global distribution properties
        - Different optimization methods may produce slightly different Z0 locations for complex distributions
        - The tolerance parameter directly affects Z0 location precision
        
    Raises:
        ValueError: If data array is empty or contains invalid values.
        ValueError: If weights array length doesn't match data array length.
        ValueError: If bounds are specified incorrectly (e.g., LB > UB).
        ValueError: If invalid parameters are provided (negative tolerance, invalid data_form, etc.).
        RuntimeError: If the fitting process fails to converge or Z0 identification fails.
        OptimizationError: If the optimization algorithm encounters numerical issues during Z0 search.
        ImportError: If required dependencies (matplotlib for plotting) are not available.
        ConvergenceError: If Z0 optimization fails to converge within specified tolerance.
        
    Comparison with Other GDF Methods:
        QLDF vs ELDF:
        - QLDF: Focuses on PDF minima (Z0 points) and local distribution around critical points
        - ELDF: Focuses on PDF maxima and local distribution around peak density regions
        
        QLDF vs QGDF/EGDF:
        - QLDF: Local analysis with detailed Z0 point characterization
        - QGDF/EGDF: Global distribution analysis across entire data range
        
        QLDF vs Traditional Methods:
        - QLDF: Advanced Z0 identification with local distribution quantification
        - KDE/Histograms: General density estimation without critical point focus
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
                 n_points: int = 500,
                 homogeneous: bool = True,
                 catch: bool = True,
                 weights: np.ndarray = None,
                 wedf: bool = False, # works better without KSDF
                 opt_method: str = 'L-BFGS-B',
                 verbose: bool = False,
                 max_data_size: int = 1000,
                 flush: bool = True):
        """
        Initialize the QLDF (Quantifying Local Distribution Function) class.

        This constructor sets up all the necessary parameters and configurations for quantifying
        local distribution characteristics and identifying Z0 points (global PDF minima) from the
        provided data. It validates input parameters and prepares the instance for subsequent
        fitting and local distribution analysis operations.

        Parameters:
            data (np.ndarray): Input data array for local distribution analysis. Must be a 1D numpy array
                             containing numerical values. Empty arrays or arrays with all NaN values
                             will raise an error. The data should ideally exhibit local distribution
                             features or critical points for meaningful QLDF analysis.
            DLB (float, optional): Data Lower Bound - the absolute minimum value that the data can
                                 theoretically take. If None, will be inferred from data. This is a
                                 hard constraint that affects Z0 identification and local distribution fitting.
            DUB (float, optional): Data Upper Bound - the absolute maximum value that the data can
                                 theoretically take. If None, will be inferred from data. This is a
                                 hard constraint that affects Z0 identification and local distribution fitting.
            LB (float, optional): Lower Probable Bound - the practical lower limit for the distribution.
                                This is typically less restrictive than DLB and helps guide Z0 search
                                within reasonable data ranges for local distribution analysis.
            UB (float, optional): Upper Probable Bound - the practical upper limit for the distribution.
                                This is typically less restrictive than DUB and helps guide Z0 search
                                within reasonable data ranges for local distribution analysis.
            S (float or str, optional): Scale parameter for the local distribution. If 'auto' (default),
                                      the scale will be automatically estimated from the data during
                                      fitting. If a float is provided, it will be used as a fixed
                                      scale parameter for local distribution characterization.
            varS (bool, optional): Whether to use variable scale parameter during optimization.
                                 Default is False. When True, allows the scale parameter to vary
                                 during fitting, which can improve local distribution characterization
                                 but may increase computational complexity.
            z0_optimize (bool, optional): Whether to optimize the Z0 location parameter during fitting.
                                        Default is True and highly recommended. When True, uses advanced
                                        optimization methods to precisely locate the global PDF minimum.
                                        Setting to False uses simple discrete minimum finding.
            tolerance (float, optional): Convergence tolerance for the optimization process, particularly
                                       for Z0 identification. Default is 1e-5. Smaller values lead to
                                       more precise Z0 location but may require more iterations.
                                       Critical for accurate local distribution analysis.
            data_form (str, optional): Form of data processing. Options are:
                                     - 'a': Additive form (default) - processes data linearly
                                     - 'm': Multiplicative form - applies log transformation for
                                            better handling of multiplicative processes in local analysis
            n_points (int, optional): Number of points to generate in the final distribution function.
                                    Higher values provide smoother curves and more precise Z0 identification
                                    but require more computation. Default is 500. Must be positive integer.
            homogeneous (bool, optional): Whether to assume data homogeneity. Default is True.
                                        Affects internal optimization strategies for local distribution fitting.
            catch (bool, optional): Whether to store intermediate calculated values during fitting.
                                  Setting to True (default) allows access to detailed results including
                                  Z0 analysis but uses more memory. Set to False for large datasets.
            weights (np.ndarray, optional): Prior weights for data points. Must be the same length
                                          as data array. If None, uniform weights (all ones) are used.
                                          Weights affect both local distribution fitting and Z0 identification.
            wedf (bool, optional): Whether to use Weighted Empirical Distribution Function in
                                 calculations. Default is False as it works better without KSDF for
                                 local distribution analysis and Z0 identification. This is a key
                                 difference from other GDF methods.
            opt_method (str, optional): Optimization method for parameter estimation and Z0 identification.
                                      Default is 'L-BFGS-B'. Other options include 'SLSQP', 'TNC', etc.
                                      Must be a valid scipy.optimize method name. Affects Z0 precision.
            verbose (bool, optional): Whether to print detailed progress information during fitting,
                                    especially Z0 optimization progress. Default is False. When True,
                                    provides diagnostic output about the optimization process and
                                    Z0 identification steps.
            max_data_size (int, optional): Maximum size of data for which smooth QLDF generation is allowed.
                                         Safety limit to prevent excessive memory usage during local
                                         distribution computation. Default is 1000.
            flush (bool, optional): Whether to flush intermediate calculations during processing.
                                  Default is True. May affect memory usage and computation speed
                                  for local distribution analysis.

        Raises:
            ValueError: If data array is empty, contains only NaN values, or has invalid dimensions.
            ValueError: If weights array is provided but has different length than data array.
            ValueError: If n_points is not a positive integer.
            ValueError: If bounds are specified incorrectly (e.g., DLB > DUB or LB > UB).
            ValueError: If data_form is not 'a' or 'm'.
            ValueError: If tolerance is not positive.
            ValueError: If max_data_size is not positive.

        Examples:
            Basic initialization for Z0 identification:
            >>> data = np.array([1, 2, 3, 4, 5])
            >>> qldf = QLDF(data)
            
            With custom bounds for guided Z0 search:
            >>> data = np.array([0.5, 1.2, 2.3, 1.8, 3.1])
            >>> weights = np.array([1.0, 0.8, 1.2, 0.9, 1.1])
            >>> qldf = QLDF(data, DLB=0, DUB=5, weights=weights, z0_optimize=True)
            
            High-precision Z0 identification:
            >>> data = np.random.bimodal_data()  # Example bimodal data
            >>> qldf = QLDF(data, tolerance=1e-8, varS=True, verbose=True)
            
            Memory-efficient setup for large datasets:
            >>> qldf = QLDF(data, catch=False, wedf=False, max_data_size=5000)
        
        Notes:
            - The initialization process does not perform any fitting or Z0 identification
            - Call fit() method afterwards to identify Z0 and perform local distribution analysis
            - Z0 optimization is highly recommended (z0_optimize=True) for accurate results
            - Bounds significantly influence Z0 identification accuracy and should be chosen carefully
            - The wedf=False default is optimized for QLDF performance and Z0 identification
            - Variable scale (varS=True) should be used when local distribution characteristics vary
            - For local distribution analysis, consider the trade-off between precision and computation time
        """
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
        Fit the Quantifying Local Distribution Function and identify the Z0 point (global PDF minimum).

        This method performs the core local distribution analysis process, including identifying the Z0 point
        where the probability density function reaches its global minimum. The fitting process focuses on
        local distribution characteristics and uses advanced optimization techniques for precise Z0 location.

        The Z0 point identification is a key feature of QLDF, representing the location of minimum probability
        density which is often critical for reliability analysis, quality control, and risk assessment applications.
        The method uses multiple advanced interpolation and optimization techniques to achieve high precision
        in Z0 identification.

        The fitting process:
            1. Preprocesses the data according to the specified data_form
            2. Sets up optimization constraints based on bounds for Z0 search
            3. Initializes parameter estimates and Z0 search domain
            4. Runs numerical optimization to find best-fit local distribution parameters
            5. Identifies Z0 point using advanced methods (spline optimization, polynomial fitting, etc.)
            6. Validates and stores the results including precise Z0 location

        Parameters:
            plot (bool, optional): Whether to automatically generate plots after fitting.
                                 Default is False. When True, creates visualization showing
                                 the fitted QLDF, PDF with identified Z0 point, and bounds.
                                 Equivalent to calling plot() method after fitting.

        Returns:
            None: The method modifies the instance in-place, storing results in self.params,
                 self.z0, and other instance attributes. Access fitted parameters via
                 self.params and Z0 location via self.z0.

        Raises:
            RuntimeError: If the optimization process fails to converge within the specified
                         tolerance and maximum iterations, or if Z0 identification fails.
            ValueError: If the data or parameters are invalid for the fitting process.
            OptimizationError: If the underlying optimization algorithm encounters numerical
                              issues during parameter estimation or Z0 search.
            ConvergenceError: If the Z0 optimization cannot find a suitable minimum location.
            ZeroIdentificationError: If Z0 point identification fails due to data characteristics.

        Side Effects:
            - Populates self.params with fitted local distribution parameters
            - Sets self.z0 with the identified global PDF minimum location
            - Updates internal state variables for local distribution analysis
            - May print progress information if verbose=True was set during initialization
            - Stores intermediate calculations if catch=True was set during initialization

        Examples:
            Basic fitting with Z0 identification:
            >>> qldf = QLDF(data)
            >>> qldf.fit()
            >>> print(f"Z0 point identified at: {qldf.z0:.6f}")
            >>> print(f"Local distribution parameters: {qldf.params}")
            
            Fitting with automatic plotting:
            >>> qldf = QLDF(data, verbose=True, z0_optimize=True)
            >>> qldf.fit(plot=True)  # Shows Z0 location on plot
            
            Monitoring Z0 optimization progress:
            >>> qldf = QLDF(data, verbose=True, tolerance=1e-8)
            >>> qldf.fit()  # Prints detailed Z0 optimization steps
            
            High-precision Z0 identification:
            >>> qldf = QLDF(bimodal_data, tolerance=1e-9, varS=True)
            >>> qldf.fit()
            >>> print(f"High-precision Z0: {qldf.z0:.10f}")

        Z0 Point Applications:
            The identified Z0 point is particularly valuable for:
            - Reliability engineering: Critical stress or load points
            - Quality control: Specification boundaries and transition zones
            - Risk assessment: Points of maximum uncertainty
            - Process optimization: Optimal operating conditions
            - Financial analysis: Market stress points or transition levels
            - Environmental monitoring: Threshold detection
            - Biostatistics: Dose-response transition points

        Quality Assessment:
            After fitting, assess Z0 identification quality by checking:
            - Z0 location is reasonable for your data context
            - Convergence status in optimization results (check self.params)
            - No warning messages during verbose output
            - Visual inspection using plot() method shows Z0 at PDF minimum
            - Multiple runs produce consistent Z0 locations

        Advanced Z0 Methods:
            QLDF uses multiple advanced methods for Z0 identification:
            1. Spline optimization with global domain search
            2. Polynomial fitting with critical point analysis
            3. Refined interpolation with high-resolution sampling
            4. Parabolic interpolation with second-derivative validation
            5. Fallback to discrete minimum if advanced methods fail

        Notes:
            - Z0 optimization is highly recommended and enabled by default
            - This method must be called before using plot() or accessing Z0 results
            - The Z0 point represents the global PDF minimum, not a local minimum
            - Multiple calls to fit() will re-run the complete optimization process
            - Z0 identification may take additional time but provides high precision
            - Failed Z0 identification may still produce valid distribution parameters
            - The quality of Z0 identification depends on data characteristics and bounds

        Troubleshooting Z0 Issues:
            If Z0 identification fails:
            - Check data for bimodal or multimodal characteristics (ideal for Z0 identification)
            - Verify bounds are reasonable and contain the expected Z0 region
            - Try different optimization methods (opt_method parameter)
            - Increase tolerance for difficult datasets
            - Use verbose=True to diagnose Z0 optimization issues
            - Ensure data has sufficient variation for meaningful Z0 identification

        Performance Considerations:
            - Z0 optimization adds computational overhead but provides high precision
            - Advanced Z0 methods are tried in order of sophistication
            - Large n_points values improve Z0 identification accuracy
            - Variable scale (varS=True) may improve Z0 identification for complex distributions
            - Memory usage increases with detailed Z0 analysis (catch=True)

        """
        self._fit_qldf(plot=plot)

    def plot(self, plot_smooth: bool = True, plot: str = 'both', bounds: bool = True, extra_df: bool = True, figsize: tuple = (12, 8)):
        """
        Visualize the fitted Quantifying Local Distribution Function with Z0 point identification.

        This method generates comprehensive visualizations of the fitted local distribution function,
        with special emphasis on the identified Z0 point (global PDF minimum) and local distribution
        characteristics. The plotting functionality provides insights into the quality of both the
        distribution fit and the Z0 identification accuracy.

        The Z0 point visualization is a key feature, showing exactly where the probability density
        reaches its global minimum, which is critical for reliability analysis, quality control,
        and understanding distribution behavior in local regions.

        Parameters:
            plot_smooth (bool, optional): Whether to plot smooth interpolated curves for the
                                        distribution functions. Default is True. When False,
                                        plots discrete points which may be useful for debugging
                                        Z0 identification or analyzing specific data points.
                                        Smooth curves provide better visualization of Z0 location.
            plot (str, optional): Type of plot to generate. Default is 'both'. Options include:
                                - 'qldf': Quantifying Local Distribution Function only
                                - 'pdf': Probability Density Function with Z0 point highlighted
                                - 'both': Both QLDF and PDF in subplots (recommended for Z0 analysis)
                                - 'all': All available plot types with comprehensive Z0 analysis
                                The 'both' option is particularly valuable for Z0 visualization.
            bounds (bool, optional): Whether to display bound lines on the plot. Default is True.
                                   Shows vertical lines for DLB, DUB, LB, and UB if specified
                                   during initialization. Critical for understanding Z0 identification
                                   constraints and validating Z0 location within expected bounds.
            extra_df (bool, optional): Whether to include additional distribution functions for
                                     comparison. Default is True. May include empirical distribution
                                     function, confidence intervals around Z0, or goodness-of-fit
                                     indicators depending on the fitting results and available data.
            figsize (tuple, optional): Figure size as (width, height) in inches. Default is (12, 8).
                                     Larger figures provide better detail for Z0 point visualization
                                     and local distribution analysis.

        Returns:
            None: The method displays the plot(s) using matplotlib. The plot window will appear
                 showing the Z0 point clearly marked on the PDF plot and local distribution
                 characteristics highlighted.

        Raises:
            RuntimeError: If fit() has not been called before plotting or if Z0 identification failed.
            ValueError: If an invalid plot type is specified.
            ImportError: If matplotlib is not available for plotting.
            PlottingError: If there are issues with the plot generation process.

        Side Effects:
            - Creates and displays matplotlib figure(s) with Z0 point highlighted
            - Z0 location is marked with vertical line and annotation
            - PDF minimum is clearly indicated with special markers
            - Updates matplotlib's current figure and axes

        Z0 Visualization Features:
            The plots include special Z0-focused elements:
            - Vertical line at Z0 location with precise coordinate annotation
            - Special marker at Z0 point on PDF curve showing global minimum
            - Z0 value displayed in legend with high precision
            - Color-coded regions showing Z0 neighborhood if applicable
            - Bounds visualization to validate Z0 location within constraints

        Examples:
            Basic Z0 visualization after fitting:
            >>> qldf = QLDF(data)
            >>> qldf.fit()
            >>> qldf.plot()  # Shows Z0 point on both QLDF and PDF plots
            
            Focus on PDF with Z0 identification:
            >>> qldf.plot(plot='pdf', bounds=True)  # PDF only with Z0 highlighted
            
            High-detail Z0 analysis:
            >>> qldf.plot(plot='both', bounds=True, extra_df=True, figsize=(16, 10))
            
            Discrete points for Z0 validation:
            >>> qldf.plot(plot_smooth=False, plot='pdf')  # Shows exact Z0 calculation points
            
            Comprehensive Z0 analysis:
            >>> qldf.plot(plot='all', bounds=True, figsize=(20, 12))  # All plots with Z0 details

        Z0 Interpretation Guide:
            Visual Z0 Assessment:
            - Z0 point should be clearly at the global PDF minimum
            - Vertical Z0 line should intersect PDF at its lowest point
            - Z0 location should be within specified bounds (if provided)
            - Local distribution should show meaningful variation around Z0
            - Multiple modes should have Z0 between them (for bimodal data)

        Quality Indicators:
            Good Z0 Identification:
            - Z0 marker precisely aligns with PDF minimum
            - Smooth PDF curve shows clear minimum at Z0
            - Z0 location is stable across different plot resolutions
            - Bounds (if shown) properly constrain Z0 location

            Poor Z0 Identification:
            - Z0 marker not at obvious PDF minimum
            - Flat or noisy PDF region around claimed Z0
            - Z0 location at boundary rather than interior minimum
            - Inconsistent Z0 location with data characteristics

        Applications for Z0 Visualization:
            - Reliability engineering: Visualizing critical failure points
            - Quality control: Identifying specification boundaries
            - Risk assessment: Understanding maximum uncertainty regions
            - Process optimization: Locating optimal operating conditions
            - Financial modeling: Identifying market stress points
            - Environmental analysis: Detecting threshold conditions

        Customization Tips:
            - Use figsize=(16, 10) or larger for detailed Z0 analysis
            - Set bounds=True when bounds were used in fitting for Z0 validation
            - Use plot='pdf' for focus on Z0 identification accuracy
            - Enable extra_df=True for comprehensive Z0 neighborhood analysis
            - Consider plot_smooth=False for debugging Z0 calculation details

        Performance Notes:
            - Z0 visualization adds minimal computational overhead
            - Smooth plots take longer but show Z0 location more clearly
            - Large n_points values (from initialization) provide better Z0 visualization
            - Complex plot types ('all') require more computation for Z0 analysis

        Notes:
            - The fit() method must be called successfully before plotting
            - Z0 location is highlighted with high precision annotation
            - Different plot types provide different perspectives on Z0 identification
            - Bounds visualization helps validate Z0 identification constraints
            - Interactive features depend on the matplotlib backend being used
            - Z0 point remains highlighted across all plot types for consistency

        Troubleshooting Z0 Visualization:
            If Z0 point appears incorrect:
            - Verify fit() completed successfully with Z0 optimization enabled
            - Check that Z0 value (qldf.z0) is reasonable for your data
            - Use verbose=True during fitting to monitor Z0 identification
            - Try different plot types to confirm Z0 location consistency
            - Validate Z0 location using bounds visualization
            - Consider refitting with different parameters if Z0 seems wrong
        """
        self._plot(plot_smooth=plot_smooth, plot=plot, bounds=bounds, extra_df=extra_df, figsize=figsize)

    def results(self) -> dict:
        """
        Retrieve the fitted parameters and comprehensive results from the QLDF fitting process with Z0 analysis.
    
        This method provides access to all key results obtained after fitting the Quantifying Local Distribution Function (QLDF) 
        to the data, with special emphasis on the identified Z0 point (global PDF minimum) and local distribution characteristics. 
        It returns a comprehensive dictionary containing fitted parameters, Z0 location analysis, local distribution properties, 
        optimization results, and diagnostic information for complete local distribution analysis.
    
        The QLDF results focus on local distribution properties and critical point identification, particularly the Z0 point
        which represents the location of minimum probability density. This makes it particularly valuable for reliability 
        analysis, critical point identification, quality control threshold analysis, and understanding localized distribution 
        behavior around points of minimum probability concentration.
    
        The results include:
            - Precise Z0 point location and identification method used
            - Local distribution bounds around critical points (DLB, DUB, LB, UB)
            - Optimal scale parameter (S_opt) for local distribution fitting
            - QLDF values representing local distribution characteristics
            - PDF values with focus on minimum regions and Z0 neighborhood
            - Z0-focused evaluation points and local distribution analysis
            - Advanced Z0 identification diagnostics and method validation
            - Optimization convergence information specific to local fitting
            - Error and warning logs from both distribution fitting and Z0 identification
    
        Returns:
            dict: A comprehensive dictionary containing fitted parameters and local distribution results.
                  Primary keys include:
                  
                  Z0 Point Analysis:
                  - 'z0': Precisely identified Z0 point location (global PDF minimum)
                  - 'z0_method': Method used for Z0 identification (e.g., 'spline_optimization', 'polynomial_fitting')
                  - 'z0_pdf_value': PDF value at the Z0 point (should be global minimum)
                  - 'z0_optimization_info': Detailed information about Z0 identification process
                  - 'z0_convergence': Convergence information for Z0 optimization
                  - 'z0_bounds_validation': Whether Z0 location satisfies specified bounds
                  
                  Local Distribution Parameters:
                  - 'DLB': Data Lower Bound used for local distribution fitting
                  - 'DUB': Data Upper Bound used for local distribution fitting  
                  - 'LB': Lower Probable Bound for local distribution analysis
                  - 'UB': Upper Probable Bound for local distribution analysis
                  - 'S_opt': Optimal scale parameter estimated for local distribution
                  - 'varS': Whether variable scale was used in local optimization
                  
                  Local Distribution Functions:
                  - 'qldf': QLDF values representing local distribution characteristics
                  - 'pdf': PDF values with detailed analysis around Z0 neighborhood
                  - 'cdf': Cumulative distribution function values (if computed)
                  - 'qldf_points': Points optimized for local distribution evaluation
                  - 'pdf_points': Points with high resolution around Z0 region
                  - 'zi': Transformed data points in local distribution domain
                  - 'zi_points': Corresponding evaluation points for local analysis
                  
                  Data and Processing Information:
                  - 'data': Original sorted input data used for local fitting
                  - 'weights': Weights applied to data points during local analysis
                  - 'wedf': Boolean indicating WEDF usage (typically False for QLDF)
                  - 'data_form': Data processing form ('a' additive, 'm' multiplicative)
                  - 'n_points': Number of points used for local distribution evaluation
                  - 'homogeneous': Data homogeneity assumption used in local fitting
                  
                  Optimization and Quality Metrics:
                  - 'fitted': Boolean confirming successful local distribution fitting
                  - 'z0_optimize': Boolean confirming Z0 optimization was enabled
                  - 'tolerance': Convergence tolerance achieved in optimization
                  - 'opt_method': Optimization method used for local parameter estimation
                  - 'max_data_size': Maximum data size limit applied during processing
                  - 'flush': Whether memory flushing was used during computation
                  
                  Advanced Diagnostics:
                  - 'warnings': List of warnings from local fitting and Z0 identification
                  - 'errors': List of errors encountered during fitting (if any)
                  - 'optimization_history': Detailed record of local optimization iterations
                  - 'z0_identification_log': Step-by-step Z0 identification process
                  - 'convergence_info': Information about local optimization convergence
                  - 'local_fit_quality': Metrics assessing local distribution fit quality
                  - 'z0_precision': Estimated precision of Z0 identification
                  - 'z0_sensitivity': Sensitivity analysis of Z0 location to parameters
    
        Raises:
            RuntimeError: If fit() has not been called before accessing results, or if Z0 identification failed.
                         The QLDF model must be successfully fitted with Z0 identification before results retrieval.
            AttributeError: If internal result structure is missing or corrupted due to fitting failure.
            KeyError: If expected result keys are unavailable, possibly due to incomplete fitting
                     or Z0 identification issues.
            ValueError: If internal state is inconsistent for result retrieval, which may occur
                       if the local distribution fitting or Z0 identification encountered issues.
            MemoryError: If results contain very large arrays that exceed available memory
                        (relevant for large datasets with high n_points values).
    
        Side Effects:
            None. This method provides read-only access to fitting results and does not modify
            the internal state of the QLDF object or trigger any recomputation of Z0 identification.
    
        Examples:
            Basic usage after local distribution fitting with Z0 identification:
            >>> qldf = QLDF(data, verbose=True, z0_optimize=True)
            >>> qldf.fit()
            >>> results = qldf.results()
            >>> print(f"Z0 point location: {results['z0']:.6f}")
            >>> print(f"Z0 identification method: {results['z0_method']}")
            >>> print(f"Local scale parameter: {results['S_opt']:.6f}")
            
            Comprehensive Z0 analysis:
            >>> z0_location = results['z0']
            >>> z0_method = results['z0_method'] 
            >>> z0_pdf_value = results['z0_pdf_value']
            >>> local_bounds = (results['LB'], results['UB'])
            >>> print(f"Z0 at {z0_location:.6f} (PDF={z0_pdf_value:.6f}) via {z0_method}")
            
            Quality assessment of Z0 identification:
            >>> if 'z0_convergence' in results:
            ...     conv_info = results['z0_convergence']
            ...     print(f"Z0 optimization converged: {conv_info.get('success', 'Unknown')}")
            >>> if results.get('z0_bounds_validation', False):
            ...     print("Z0 location validated within specified bounds")
            >>> if results['warnings']:
            ...     print(f"Z0 identification warnings: {len(results['warnings'])}")
            
            Advanced Z0 diagnostics:
            >>> if 'z0_identification_log' in results:
            ...     log = results['z0_identification_log']
            ...     print(f"Z0 identification attempts: {len(log)}")
            ...     for entry in log:
            ...         method = entry.get('method', 'unknown')
            ...         success = entry.get('success', False)
            ...         print(f"  {method}: {'Success' if success else 'Failed'}")
            
            Local distribution analysis:
            >>> local_pdf = results['pdf']
            >>> z0_index = np.argmin(local_pdf)  # Should correspond to Z0 location
            >>> zi_points = results['zi_points']
            >>> print(f"Z0 verification: PDF minimum at zi={zi_points[z0_index]:.6f}")
            
            Optimization performance analysis:
            >>> if results.get('optimization_history'):
            ...     history = results['optimization_history']
            ...     print(f"Local optimization iterations: {len(history)}")
            ...     if history:
            ...         final_loss = history[-1].get('total_loss', 'N/A')
            ...         print(f"Final optimization loss: {final_loss}")
    
        Applications of QLDF Results:
            Z0 Point Applications:
            - Reliability engineering: Critical failure points and stress thresholds
            - Quality control: Specification boundaries and acceptable limit determination
            - Risk assessment: Points of maximum uncertainty and transition analysis
            - Process optimization: Optimal operating points with minimum variation
            - Financial modeling: Market stress points and volatility transitions
            - Environmental monitoring: Threshold detection and critical limit analysis
            - Biostatistics: Dose-response transition points and therapeutic windows
            - Signal processing: Discontinuity detection and change point analysis
            
            Local Distribution Applications:
            - Neighborhood analysis around critical points (Z0 region)
            - Local probability density characterization for risk assessment
            - Threshold-based decision making using Z0 location
            - Quality control chart development with Z0-based control limits
            - Failure mode analysis with Z0 as critical operating point
    
        Interpretation Guide:
            Z0 Point Results:
            - 'z0': The precise location where PDF reaches global minimum
            - 'z0_method': Indicates reliability of identification ('spline_optimization' > 'discrete_extremum')
            - 'z0_pdf_value': Should be the smallest value in the PDF array
            - Low 'z0_pdf_value' indicates well-defined minimum region
            
            Local Distribution Parameters:
            - 'S_opt': Scale parameter controls local distribution spread around Z0
            - Bounds (LB, UB): Define effective local analysis range around Z0
            - 'varS=True' results may show variable scale adaptation for local regions
            
            Quality Indicators:
            - Empty 'warnings' and 'errors' indicate successful Z0 identification
            - 'z0_convergence.success=True' confirms reliable Z0 optimization
            - 'z0_bounds_validation=True' indicates Z0 within reasonable constraints
            - Advanced 'z0_method' (not 'discrete_extremum') suggests high precision
    
        Performance Considerations:
            - Results retrieval is immediate and optimized for Z0 analysis access
            - Z0 identification adds diagnostic information to standard fitting results
            - Large n_points values increase Z0 precision but also result array sizes
            - Local distribution evaluation focuses computational resources around Z0 region
            - Advanced Z0 methods generate more detailed diagnostic information
            - Memory usage scales with both data size and Z0 analysis detail level
    
        Comparison with Other GDF Results:
            QLDF vs ELDF Results:
            - QLDF: Z0 point (global PDF minimum) identification and local minimum analysis
            - ELDF: Z0 point (global PDF maximum) identification and local maximum analysis
            
            QLDF vs QGDF/EGDF Results:
            - QLDF: Local distribution focus with detailed Z0 critical point analysis
            - QGDF/EGDF: Global distribution characteristics across entire data range
            
            QLDF Unique Features:
            - 'z0_method' indicates advanced Z0 identification technique used
            - 'z0_identification_log' provides step-by-step Z0 optimization history
            - 'z0_precision' and 'z0_sensitivity' offer Z0 reliability assessment
            - Local distribution focus rather than global distribution properties
    
        Notes:
            - fit() with z0_optimize=True must complete successfully before accessing Z0 results
            - Z0 identification may generate extensive diagnostic information for complex data
            - Results structure is consistent regardless of Z0 identification method succeeded
            - Advanced Z0 methods (spline, polynomial) provide more detailed diagnostics
            - Local distribution focus means results emphasize Z0 neighborhood characteristics
            - If catch=False was used, some detailed Z0 intermediate results may be unavailable
            - Z0 location represents global PDF minimum, not local minima
            - All numeric results use high precision appropriate for Z0 analysis
            - Results can be serialized for reproducible Z0 analysis and reporting
    
        Troubleshooting:
            Incomplete Z0 Results:
            - Verify fit() completed without exceptions and Z0 optimization was enabled
            - Check 'z0_identification_log' for specific Z0 identification failures
            - Ensure adequate tolerance for Z0 optimization convergence
            
            Unexpected Z0 Location:
            - Cross-reference 'z0' with PDF values to confirm it's at global minimum
            - Check 'z0_bounds_validation' for constraint satisfaction
            - Review 'z0_method' - advanced methods are more reliable than discrete
            - Examine 'z0_sensitivity' if available for Z0 location stability
            
            Poor Z0 Identification Quality:
            - Check 'warnings' for insights into Z0 identification challenges
            - Verify data has sufficient variation for meaningful Z0 identification
            - Consider tighter tolerance or different optimization method for Z0
            - Ensure bounds appropriately constrain Z0 search region
            
            Z0 Optimization Issues:
            - Review 'z0_convergence' information for optimization problems
            - Check 'z0_identification_log' for method-specific failure patterns
            - Consider enabling verbose=True during fitting for Z0 diagnostic output
            - Try different data preprocessing or bound specifications for Z0 search
        """
        if not self._fitted:
            raise RuntimeError("Must fit QLDF before getting results.")
        
        return self._get_results()