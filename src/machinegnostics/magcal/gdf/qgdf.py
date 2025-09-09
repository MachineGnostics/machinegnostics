'''
QGDF Quantifying Global Distribution Function (QGDF)

Public class

Author: Nirmal Parmar
Machine Gnostics
'''

import numpy as np
from machinegnostics.magcal.gdf.base_qgdf import BaseQGDF

class QGDF(BaseQGDF):
    """
    QGDF - Quantifying Global Distribution Function.
    
    A comprehensive class for quantifying and analyzing global distribution characteristics across entire datasets with 
    focus on identifying and analyzing points of minimum probability density (Z0 points) and their global impact on 
    distribution behavior. This class provides methods to fit global distribution functions that characterize the 
    complete probabilistic structure of data while maintaining focus on critical minimum points.

    The QGDF class specializes in global distribution analysis, making it particularly valuable for understanding 
    overall distribution behavior, global probability density patterns, and comprehensive distribution modeling. 
    Unlike local distribution methods, QGDF provides complete distributional characterization across the entire 
    data domain while identifying globally significant critical points (Z0).

    The Quantifying Global Distribution Function (QGDF) is a gnostic-probabilistic model that identifies and quantifies 
    global distribution characteristics, with particular emphasis on regions of low probability density that have 
    global significance. It uses advanced optimization techniques to locate Z0 points (global PDF minima) and provides 
    comprehensive analysis of how these critical points influence the entire distribution structure.

    Key Features:
        - Automatic Z0 point identification (global PDF minimum) with global impact analysis
        - Complete global distribution characterization across entire data range
        - Advanced interpolation methods for precise Z0 estimation with global validation
        - Support for weighted data analysis with global weight distribution
        - Memory-efficient processing for large datasets with global scope
        - Comprehensive visualization of global distribution features with Z0 highlighting
        - Robust optimization with multiple solver options for global parameter estimation
        - Integration with empirical distribution methods for global distribution fitting

    Attributes:
        data (np.ndarray): The input dataset used for global distribution analysis.
        DLB (float): Data Lower Bound - absolute minimum value the data can take globally.
        DUB (float): Data Upper Bound - absolute maximum value the data can take globally.
        LB (float): Lower Probable Bound - practical lower limit for global distribution.
        UB (float): Upper Probable Bound - practical upper limit for global distribution.
        S (float or str): Scale parameter for global distribution. Set to 'auto' for automatic estimation.
        z0_optimize (bool): Whether to optimize the location parameter z0 during global fitting (default: True).
        data_form (str): Form of the data processing:
            - 'a': Additive form (default) - treats data linearly across global range
            - 'm': Multiplicative form - applies logarithmic transformation for global analysis
        n_points (int): Number of points to generate in the global distribution function (default: 500).
        catch (bool): Whether to store intermediate calculated values during global fitting (default: True).
        weights (np.ndarray): Prior weights for data points in global analysis. If None, uniform weights are used.
        wedf (bool): Whether to use Weighted Empirical Distribution Function (default: False for better performance).
        opt_method (str): Optimization method for global parameter estimation (default: 'L-BFGS-B').
        tolerance (float): Convergence tolerance for global optimization (default: 1e-9).
        verbose (bool): Whether to print detailed progress information during global fitting (default: False).
        params (dict): Dictionary storing fitted global parameters and results after fitting.
        homogeneous (bool): To indicate data homogeneity for global analysis (default: True).
        max_data_size (int): Maximum data size for smooth global QGDF generation (default: 1000).
        flush (bool): Whether to flush large arrays during global computation (default: True).
        z0 (float): Identified Z0 point (global PDF minimum) after fitting with global significance.

    Methods:
        fit(): Fit the Quantifying Global Distribution Function to the entire dataset.
        plot(plot_smooth=True, plot='both', bounds=True, extra_df=True, figsize=(12,8)): 
            Visualize the fitted global distribution with Z0 point identification and global context.
        results(): Get comprehensive results including Z0 location and global distribution parameters.
    
    Examples:
        Basic usage with automatic global Z0 identification:
        >>> import numpy as np
        >>> from machinegnostics.magcal import QGDF
        >>> 
        >>> # Multi-modal data with global minimum between major modes
        >>> data = np.concatenate([np.random.normal(-3, 0.8, 200), 
        ...                       np.random.normal(0, 0.3, 50),
        ...                       np.random.normal(3, 0.8, 200)])
        >>> qgdf = QGDF(data)
        >>> qgdf.fit()
        >>> print(f"Global Z0 point (PDF minimum): {qgdf.z0:.4f}")
        >>> qgdf.plot()
        
        Usage with custom global bounds and high-precision optimization:
        >>> # Global financial market data analysis
        >>> market_data = np.array([-0.08, -0.05, -0.02, 0.01, 0.03, -0.01, 0.02, 0.04, -0.03, 0.06])
        >>> qgdf = QGDF(market_data, LB=-0.15, UB=0.15, z0_optimize=True, tolerance=1e-9)
        >>> qgdf.fit()
        >>> results = qgdf.results()
        >>> print(f"Global critical point at: {results['z0']:.8f}")
        >>> qgdf.plot(bounds=True)
        
        Advanced global Z0 analysis with comprehensive scope:
        >>> # Quality control data with global specification analysis
        >>> measurements = np.concatenate([
        ...     np.random.gamma(2, 2, 300),  # Main process
        ...     np.random.gamma(0.5, 1, 50), # Defective items
        ...     np.random.gamma(4, 1, 100)   # Over-specification
        ... ])
        >>> qgdf = QGDF(measurements, 
        ...              DLB=0, DUB=20,
        ...              tolerance=1e-8,
        ...              verbose=True)
        >>> qgdf.fit()
        >>> qgdf.plot(plot='both', bounds=True)
        
        Memory-efficient global processing for very large datasets:
        >>> # Large-scale global dataset analysis
        >>> large_global_data = np.concatenate([
        ...     np.random.weibull(1.5, 50000),    # Primary distribution
        ...     np.random.weibull(0.8, 20000),    # Secondary mode
        ...     np.random.exponential(2, 30000)    # Tail behavior
        ... ])
        >>> qgdf = QGDF(large_global_data,
        ...              catch=False,  # Save memory for global analysis
        ...              n_points=1000,  # Higher resolution for global view
        ...              max_data_size=50000)
        >>> qgdf.fit()
        >>> print(f"Global Z0 at PDF minimum: {qgdf.z0:.6f}")
    
    Workflow:
        1. Initialize QGDF with your data and desired global parameters
        2. Call fit() to identify global Z0 point and estimate comprehensive distribution parameters
        3. Use plot() to visualize Z0 location in global context and complete distribution characteristics
        4. Access detailed results for global analysis and decision making
        
        >>> qgdf = QGDF(data, DLB=0, UB=1000)  # Step 1: Initialize for global analysis
        >>> qgdf.fit()                         # Step 2: Fit globally and find Z0
        >>> qgdf.plot(plot='both')            # Step 3: Visualize global context
        >>> results = qgdf.results()          # Step 4: Get comprehensive global results
    
    Performance Tips:
        - Set wedf=False for better performance in global analysis (default recommendation)
        - Use higher n_points values for better global resolution and Z0 identification
        - Set appropriate global bounds to improve Z0 identification across entire range
        - Use catch=False for very large datasets to save memory during global computation
        - Adjust tolerance based on required global Z0 precision (default 1e-9 for high precision)
        - Use verbose=True to monitor global Z0 optimization progress
        - For repeated global analysis, save fitted parameters and Z0 location
    
    Common Use Cases:
        - System-wide reliability analysis (identifying global failure modes)
        - Market-wide financial analysis (global risk assessment and stress testing)
        - Population health studies (global health trend analysis with critical thresholds)
        - Manufacturing process optimization (global quality control with specification limits)
        - Environmental monitoring (global ecosystem analysis with critical transition points)
        - Supply chain analysis (global bottleneck identification and risk points)
        - Network performance analysis (global system stress points and capacity limits)
        - Economic modeling (global market equilibrium points and stability analysis)
    
    Global Z0 Point Applications:
        The globally identified Z0 point is particularly valuable for:
        - System-wide risk management: Global points of maximum uncertainty or instability
        - Enterprise quality control: Global specification boundaries affecting entire production
        - Market analysis: Global equilibrium points with minimum market activity or maximum stress
        - Reliability engineering: Global stress levels affecting entire system performance
        - Environmental policy: Global threshold levels for regulatory and policy decisions
        - Scientific research: Global phase transitions and critical phenomena across systems
    
    Notes:
        - Global Z0 optimization is enabled by default and essential for comprehensive analysis
        - The wedf parameter is set to False by default as it works better for global QGDF analysis
        - Global bounds (DLB, DUB, LB, UB) significantly influence Z0 identification across entire range
        - Higher precision tolerance (1e-9) ensures accurate global Z0 identification
        - The global Z0 point represents the absolute minimum of the PDF across entire distribution
        - QGDF focuses on complete distribution characteristics rather than local features only
        - Global optimization methods may require more computation but provide comprehensive results
        - The tolerance parameter directly affects global Z0 location precision across entire domain
        
    Raises:
        ValueError: If data array is empty or contains invalid values for global analysis.
        ValueError: If weights array length doesn't match data array length for global computation.
        ValueError: If bounds are specified incorrectly for global analysis (e.g., LB > UB).
        ValueError: If invalid parameters are provided for global fitting (negative tolerance, invalid data_form, etc.).
        RuntimeError: If the global fitting process fails to converge or global Z0 identification fails.
        OptimizationError: If the optimization algorithm encounters numerical issues during global Z0 search.
        ImportError: If required dependencies (matplotlib for plotting) are not available.
        ConvergenceError: If global Z0 optimization fails to converge within specified tolerance.
        
    Comparison with Other GDF Methods:
        QGDF vs QLDF:
        - QGDF: Focuses on PDF minima (Z0 points) with global distribution analysis across entire data range
        - QLDF: Focuses on PDF minima (Z0 points) with local distribution around critical points only
        
        QGDF vs EGDF/ELDF:
        - QGDF: Global analysis with Z0 point (PDF minimum) identification and complete distribution modeling
        - EGDF/ELDF: Global/local analysis with focus on PDF maxima rather than minima
        
        QGDF vs Traditional Methods:
        - QGDF: Advanced global Z0 identification with comprehensive distribution quantification
        - KDE/Histograms: General density estimation without critical point focus or global optimization
    """

    def __init__(self,
                 data: np.ndarray,
                 DLB: float = None,
                 DUB: float = None,
                 LB: float = None,
                 UB: float = None,
                 S = 'auto',
                 z0_optimize: bool = True,
                 tolerance: float = 1e-9,
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
        Initialize the QGDF (Quantifying Global Distribution Function) class.

        This constructor sets up all the necessary parameters and configurations for quantifying
        global distribution characteristics and identifying Z0 points (global PDF minima) across
        the entire data range. It validates input parameters and prepares the instance for subsequent
        global fitting and comprehensive distribution analysis operations.

        Parameters:
            data (np.ndarray): Input data array for global distribution analysis. Must be a 1D numpy array
                             containing numerical values. Empty arrays or arrays with all NaN values
                             will raise an error. The data should represent the complete population or
                             sample for meaningful global QGDF analysis and Z0 identification.
            DLB (float, optional): Data Lower Bound - the absolute minimum value that the data can
                                 theoretically take across the global range. If None, will be inferred from data. 
                                 This is a hard constraint that affects global Z0 identification and comprehensive 
                                 distribution fitting across the entire domain.
            DUB (float, optional): Data Upper Bound - the absolute maximum value that the data can
                                 theoretically take across the global range. If None, will be inferred from data. 
                                 This is a hard constraint that affects global Z0 identification and comprehensive 
                                 distribution fitting across the entire domain.
            LB (float, optional): Lower Probable Bound - the practical lower limit for global distribution.
                                This is typically less restrictive than DLB and helps guide global Z0 search
                                within reasonable data ranges for comprehensive distribution analysis.
            UB (float, optional): Upper Probable Bound - the practical upper limit for global distribution.
                                This is typically less restrictive than DUB and helps guide global Z0 search
                                within reasonable data ranges for comprehensive distribution analysis.
            S (float or str, optional): Scale parameter for the global distribution. If 'auto' (default),
                                      the scale will be automatically estimated from the entire data during
                                      global fitting. If a float is provided, it will be used as a fixed
                                      scale parameter for global distribution characterization.
            z0_optimize (bool, optional): Whether to optimize the Z0 location parameter during global fitting.
                                        Default is True and essential for comprehensive global analysis. When True, 
                                        uses advanced optimization methods to precisely locate the global PDF minimum
                                        across the entire data range. Setting to False uses simple discrete minimum 
                                        finding which may miss global optimum.
            tolerance (float, optional): Convergence tolerance for the global optimization process, particularly
                                       for Z0 identification across entire domain. Default is 1e-9 for high precision
                                       global analysis. Smaller values lead to more precise global Z0 location but may 
                                       require more iterations. Critical for accurate global distribution analysis.
            data_form (str, optional): Form of data processing for global analysis. Options are:
                                     - 'a': Additive form (default) - processes data linearly across global range
                                     - 'm': Multiplicative form - applies log transformation for better handling 
                                            of multiplicative processes in global analysis
            n_points (int, optional): Number of points to generate in the final global distribution function.
                                    Higher values provide smoother curves and more precise global Z0 identification
                                    but require more computation. Default is 500. For global analysis, consider
                                    higher values (1000+) for better resolution. Must be positive integer.
            homogeneous (bool, optional): Whether to assume data homogeneity for global analysis. Default is True.
                                        Affects internal optimization strategies for global distribution fitting
                                        and Z0 identification across the entire data range.
            catch (bool, optional): Whether to store intermediate calculated values during global fitting.
                                  Setting to True (default) allows access to detailed results including
                                  comprehensive Z0 analysis but uses more memory. Set to False for very large 
                                  datasets requiring global analysis.
            weights (np.ndarray, optional): Prior weights for data points in global analysis. Must be the same length
                                          as data array. If None, uniform weights (all ones) are used.
                                          Weights affect both global distribution fitting and Z0 identification
                                          across the entire data range.
            wedf (bool, optional): Whether to use Weighted Empirical Distribution Function in global
                                 calculations. Default is False as it works better without KSDF for
                                 global distribution analysis and Z0 identification. This is a key
                                 optimization for QGDF performance in global analysis.
            opt_method (str, optional): Optimization method for global parameter estimation and Z0 identification.
                                      Default is 'L-BFGS-B'. Other options include 'SLSQP', 'TNC', etc.
                                      Must be a valid scipy.optimize method name. Affects global Z0 precision
                                      and convergence across entire data domain.
            verbose (bool, optional): Whether to print detailed progress information during global fitting,
                                    especially Z0 optimization progress across entire range. Default is False. 
                                    When True, provides diagnostic output about the global optimization process 
                                    and Z0 identification steps.
            max_data_size (int, optional): Maximum size of data for which smooth global QGDF generation is allowed.
                                         Safety limit to prevent excessive memory usage during global
                                         distribution computation. Default is 1000. For global analysis of
                                         larger datasets, consider increasing this value.
            flush (bool, optional): Whether to flush intermediate calculations during global processing.
                                  Default is True. May affect memory usage and computation speed
                                  for global distribution analysis across large data ranges.

        Raises:
            ValueError: If data array is empty, contains only NaN values, or has invalid dimensions for global analysis.
            ValueError: If weights array is provided but has different length than data array for global computation.
            ValueError: If n_points is not a positive integer suitable for global analysis.
            ValueError: If bounds are specified incorrectly for global analysis (e.g., DLB > DUB or LB > UB).
            ValueError: If data_form is not 'a' or 'm' for global processing.
            ValueError: If tolerance is not positive for global optimization.
            ValueError: If max_data_size is not positive for global computation limits.

        Examples:
            Basic initialization for global Z0 identification:
            >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            >>> qgdf = QGDF(data)
            
            With custom global bounds for comprehensive Z0 search:
            >>> data = np.array([0.5, 1.2, 2.3, 1.8, 3.1, 2.7, 4.2, 3.8, 5.1])
            >>> weights = np.array([1.0, 0.8, 1.2, 0.9, 1.1, 1.0, 0.7, 1.3, 0.9])
            >>> qgdf = QGDF(data, DLB=0, DUB=10, weights=weights, z0_optimize=True)
            
            High-precision global Z0 identification:
            >>> multimodal_data = np.concatenate([np.random.normal(-2, 0.5, 100),
            ...                                  np.random.normal(2, 0.5, 100)])
            >>> qgdf = QGDF(multimodal_data, tolerance=1e-10, verbose=True)
            
            Memory-efficient setup for large global datasets:
            >>> qgdf = QGDF(large_data, catch=False, wedf=False, 
            ...             max_data_size=10000, n_points=1000)
        
        Notes:
            - The initialization process does not perform any fitting or global Z0 identification
            - Call fit() method afterwards to identify global Z0 and perform comprehensive distribution analysis
            - Global Z0 optimization is highly recommended (z0_optimize=True) for accurate results
            - Global bounds significantly influence Z0 identification accuracy across entire data range
            - The wedf=False default is optimized for QGDF performance in global analysis
            - Higher tolerance precision (1e-9) is used by default for accurate global Z0 identification
            - For global distribution analysis, consider the trade-off between precision and computation time
            - Global analysis requires consideration of entire data range, not just local features
        """
        super().__init__(data=data,
                         DLB=DLB,
                         DUB=DUB,
                         LB=LB,
                         UB=UB,
                         S=S,
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
        Fit the Quantifying Global Distribution Function and identify the global Z0 point (global PDF minimum).

        This method performs the core global distribution analysis process, including identifying the Z0 point
        where the probability density function reaches its global minimum across the entire data range. The fitting 
        process focuses on comprehensive distribution characteristics and uses advanced optimization techniques for 
        precise global Z0 location with validation across the complete data domain.

        The global Z0 point identification is a key feature of QGDF, representing the location of minimum probability
        density that has significance across the entire distribution. This is critical for system-wide reliability 
        analysis, global risk assessment, comprehensive quality control, and understanding distribution behavior 
        that affects the entire data range rather than local regions only.

        The global fitting process:
            1. Preprocesses the data according to the specified data_form across entire range
            2. Sets up global optimization constraints based on bounds for comprehensive Z0 search
            3. Initializes parameter estimates and Z0 search domain across complete data range
            4. Runs numerical optimization to find best-fit global distribution parameters
            5. Identifies global Z0 point using advanced methods with validation across entire domain
            6. Validates and stores the results including precise global Z0 location and impact analysis

        Parameters:
            plot (bool, optional): Whether to automatically generate plots after global fitting.
                                 Default is False. When True, creates comprehensive visualization showing
                                 the fitted global QGDF, PDF with identified global Z0 point, and bounds
                                 across entire data range. Equivalent to calling plot() method after fitting.

        Returns:
            None: The method modifies the instance in-place, storing results in self.params,
                 self.z0, and other instance attributes. Access fitted global parameters via
                 self.params and global Z0 location via self.z0.

        Raises:
            RuntimeError: If the global optimization process fails to converge within the specified
                         tolerance and maximum iterations, or if global Z0 identification fails.
            ValueError: If the data or parameters are invalid for global fitting process.
            OptimizationError: If the underlying optimization algorithm encounters numerical
                              issues during global parameter estimation or Z0 search.
            ConvergenceError: If the global Z0 optimization cannot find a suitable minimum location
                             across the entire data range.
            GlobalOptimizationError: If global Z0 point identification fails due to data characteristics
                                   or insufficient global variation for meaningful analysis.

        Side Effects:
            - Populates self.params with fitted global distribution parameters
            - Sets self.z0 with the identified global PDF minimum location
            - Updates internal state variables for comprehensive global distribution analysis
            - May print progress information if verbose=True was set during initialization
            - Stores intermediate calculations if catch=True was set during initialization

        Examples:
            Basic global fitting with Z0 identification:
            >>> qgdf = QGDF(data)
            >>> qgdf.fit()
            >>> print(f"Global Z0 point identified at: {qgdf.z0:.6f}")
            >>> print(f"Global distribution parameters: {qgdf.params}")
            
            Global fitting with automatic comprehensive plotting:
            >>> qgdf = QGDF(data, verbose=True, z0_optimize=True, tolerance=1e-9)
            >>> qgdf.fit(plot=True)  # Shows global Z0 location on comprehensive plot
            
            Monitoring global Z0 optimization progress:
            >>> qgdf = QGDF(data, verbose=True, tolerance=1e-10)
            >>> qgdf.fit()  # Prints detailed global Z0 optimization steps
            
            High-precision global Z0 identification:
            >>> qgdf = QGDF(multimodal_global_data, tolerance=1e-11, n_points=1000)
            >>> qgdf.fit()
            >>> print(f"High-precision global Z0: {qgdf.z0:.12f}")

        Global Z0 Point Applications:
            The identified global Z0 point is particularly valuable for:
            - System-wide reliability engineering: Global critical stress or failure points
            - Enterprise quality control: Global specification boundaries affecting entire production
            - Market-wide risk assessment: Global equilibrium points with minimum market activity
            - Environmental policy: Global threshold levels for regulatory decisions
            - Supply chain optimization: Global bottleneck identification and critical points
            - Network performance: Global system stress points and capacity limits
            - Scientific research: Global phase transitions and critical phenomena

        Quality Assessment:
            After global fitting, assess Z0 identification quality by checking:
            - Global Z0 location is reasonable for entire data context
            - Convergence status in global optimization results (check self.params)
            - No warning messages during verbose global optimization output
            - Visual inspection using plot() method shows Z0 at global PDF minimum
            - Multiple global runs produce consistent Z0 locations across entire range

        Advanced Global Z0 Methods:
            QGDF uses multiple advanced methods for global Z0 identification:
            1. Global spline optimization with complete domain search
            2. Global polynomial fitting with critical point analysis across entire range
            3. Refined interpolation with high-resolution sampling across complete domain
            4. Parabolic interpolation with second-derivative validation globally
            5. Fallback to discrete global minimum if advanced methods fail

        Notes:
            - Global Z0 optimization is highly recommended and enabled by default
            - This method must be called before using plot() or accessing global Z0 results
            - The global Z0 point represents the absolute PDF minimum across entire data range
            - Multiple calls to fit() will re-run the complete global optimization process
            - Global Z0 identification may take additional time but provides comprehensive precision
            - Failed global Z0 identification may still produce valid global distribution parameters
            - The quality of global Z0 identification depends on data characteristics and global bounds

        Troubleshooting Global Z0 Issues:
            If global Z0 identification fails:
            - Check data for multimodal characteristics suitable for global Z0 identification
            - Verify global bounds encompass the expected Z0 region across entire range
            - Try different optimization methods (opt_method parameter) for global search
            - Increase tolerance precision for difficult global datasets
            - Use verbose=True to diagnose global Z0 optimization issues
            - Ensure data has sufficient global variation for meaningful Z0 identification

        Performance Considerations:
            - Global Z0 optimization adds computational overhead but provides comprehensive precision
            - Advanced global Z0 methods are tried in order of sophistication across entire domain
            - Large n_points values improve global Z0 identification accuracy across complete range
            - Higher precision tolerance (1e-9 default) ensures accurate global Z0 identification
            - Memory usage increases with detailed global Z0 analysis (catch=True)
            - Global analysis requires more computation than local methods but provides complete picture

        """
        self._fit_qgdf(plot=plot)

    def plot(self, plot_smooth: bool = True, plot: str = 'both', bounds: bool = True, extra_df: bool = True, figsize: tuple = (12, 8)):
        """
        Visualize the fitted Quantifying Global Distribution Function with global Z0 point identification.

        This method generates comprehensive visualizations of the fitted global distribution function,
        with special emphasis on the identified global Z0 point (global PDF minimum) and complete distribution
        characteristics across the entire data range. The plotting functionality provides insights into the quality 
        of both the global distribution fit and the Z0 identification accuracy within the complete data context.

        The global Z0 point visualization is a key feature, showing exactly where the probability density
        reaches its global minimum across the entire data range, which is critical for system-wide reliability 
        analysis, comprehensive quality control, and understanding distribution behavior that affects the 
        complete data domain rather than local regions only.

        Parameters:
            plot_smooth (bool, optional): Whether to plot smooth interpolated curves for the
                                        global distribution functions. Default is True. When False,
                                        plots discrete points which may be useful for debugging
                                        global Z0 identification or analyzing specific data points
                                        across entire range. Smooth curves provide better visualization 
                                        of global Z0 location and complete distribution characteristics.
            plot (str, optional): Type of plot to generate for global analysis. Default is 'both'. Options include:
                                - 'qgdf': Quantifying Global Distribution Function only across entire range
                                - 'pdf': Probability Density Function with global Z0 point highlighted
                                - 'both': Both QGDF and PDF in subplots (recommended for global Z0 analysis)
                                - 'all': All available plot types with comprehensive global Z0 analysis
                                The 'both' option is particularly valuable for global Z0 visualization 
                                and understanding complete distribution context.
            bounds (bool, optional): Whether to display bound lines on the global plot. Default is True.
                                   Shows vertical lines for DLB, DUB, LB, and UB if specified
                                   during initialization. Critical for understanding global Z0 identification
                                   constraints and validating Z0 location within expected global bounds.
            extra_df (bool, optional): Whether to include additional distribution functions for
                                     global comparison. Default is True. May include empirical distribution
                                     function, confidence intervals around global Z0, or goodness-of-fit
                                     indicators depending on the global fitting results and available data
                                     across the entire range.
            figsize (tuple, optional): Figure size as (width, height) in inches. Default is (12, 8).
                                     Larger figures provide better detail for global Z0 point visualization
                                     and comprehensive distribution analysis across entire data range.

        Returns:
            None: The method displays the plot(s) using matplotlib. The plot window will appear
                 showing the global Z0 point clearly marked on the PDF plot and complete distribution
                 characteristics highlighted across the entire data range.

        Raises:
            RuntimeError: If fit() has not been called before plotting or if global Z0 identification failed.
            ValueError: If an invalid plot type is specified for global visualization.
            ImportError: If matplotlib is not available for plotting.
            PlottingError: If there are issues with the global plot generation process.

        Side Effects:
            - Creates and displays matplotlib figure(s) with global Z0 point highlighted
            - Global Z0 location is marked with vertical line and annotation across entire range
            - Global PDF minimum is clearly indicated with special markers
            - Updates matplotlib's current figure and axes with global context

        Global Z0 Visualization Features:
            The plots include special global Z0-focused elements:
            - Vertical line at global Z0 location with precise coordinate annotation
            - Special marker at global Z0 point on PDF curve showing absolute minimum
            - Global Z0 value displayed in legend with high precision
            - Color-coded regions showing global Z0 significance across entire range
            - Bounds visualization to validate global Z0 location within complete data constraints

        Examples:
            Basic global Z0 visualization after fitting:
            >>> qgdf = QGDF(data)
            >>> qgdf.fit()
            >>> qgdf.plot()  # Shows global Z0 point on both QGDF and PDF plots
            
            Focus on PDF with global Z0 identification:
            >>> qgdf.plot(plot='pdf', bounds=True)  # PDF only with global Z0 highlighted
            
            High-detail global Z0 analysis:
            >>> qgdf.plot(plot='both', bounds=True, extra_df=True, figsize=(16, 10))
            
            Discrete points for global Z0 validation:
            >>> qgdf.plot(plot_smooth=False, plot='pdf')  # Shows exact global Z0 calculation points
            
            Comprehensive global Z0 analysis:
            >>> qgdf.plot(plot='all', bounds=True, figsize=(20, 12))  # All plots with global Z0 details

        Global Z0 Interpretation Guide:
            Visual Global Z0 Assessment:
            - Global Z0 point should be clearly at the absolute PDF minimum across entire range
            - Vertical global Z0 line should intersect PDF at its lowest point globally
            - Global Z0 location should be within specified bounds (if provided)
            - Global distribution should show meaningful variation across complete range
            - Multiple modes should have global Z0 at the absolute minimum between them

        Quality Indicators:
            Good Global Z0 Identification:
            - Global Z0 marker precisely aligns with absolute PDF minimum
            - Smooth PDF curve shows clear global minimum at Z0 across entire range
            - Global Z0 location is stable across different plot resolutions
            - Bounds (if shown) properly constrain global Z0 location within complete domain

            Poor Global Z0 Identification:
            - Global Z0 marker not at obvious absolute PDF minimum
            - Flat or noisy PDF region around claimed global Z0
            - Global Z0 location at boundary rather than interior global minimum
            - Inconsistent global Z0 location with complete data characteristics

        Applications for Global Z0 Visualization:
            - System-wide reliability engineering: Visualizing global critical failure points
            - Enterprise quality control: Identifying global specification boundaries
            - Market-wide risk assessment: Understanding global equilibrium and stress regions
            - Environmental policy: Locating global threshold conditions for regulations
            - Supply chain optimization: Identifying global bottleneck and critical points
            - Network performance: Detecting global system stress points and capacity limits

        Customization Tips:
            - Use figsize=(16, 10) or larger for detailed global Z0 analysis
            - Set bounds=True when bounds were used in fitting for global Z0 validation
            - Use plot='pdf' for focus on global Z0 identification accuracy
            - Enable extra_df=True for comprehensive global Z0 analysis across entire range
            - Consider plot_smooth=False for debugging global Z0 calculation details

        Performance Notes:
            - Global Z0 visualization adds minimal computational overhead
            - Smooth plots take longer but show global Z0 location more clearly
            - Large n_points values (from initialization) provide better global Z0 visualization
            - Complex plot types ('all') require more computation for comprehensive global analysis

        Notes:
            - The fit() method must be called successfully before plotting
            - Global Z0 location is highlighted with high precision annotation
            - Different plot types provide different perspectives on global Z0 identification
            - Bounds visualization helps validate global Z0 identification constraints
            - Interactive features depend on the matplotlib backend being used
            - Global Z0 point remains highlighted across all plot types for consistency

        Troubleshooting Global Z0 Visualization:
            If global Z0 point appears incorrect:
            - Verify fit() completed successfully with global Z0 optimization enabled
            - Check that global Z0 value (qgdf.z0) is reasonable for complete data context
            - Use verbose=True during fitting to monitor global Z0 identification
            - Try different plot types to confirm global Z0 location consistency
            - Validate global Z0 location using bounds visualization across entire range
            - Consider refitting with different global parameters if Z0 seems wrong
        """
        self._plot(plot_smooth=plot_smooth, plot=plot, bounds=bounds, extra_df=extra_df, figsize=figsize)
    
    def results(self) -> dict:
        """
        Retrieve the fitted parameters and comprehensive results from the global QGDF fitting process with Z0 analysis.
    
        This method provides access to all key results obtained after fitting the Quantifying Global Distribution Function (QGDF) 
        to the data, with special emphasis on the identified global Z0 point (global PDF minimum) and comprehensive distribution 
        characteristics across the entire data range. It returns a comprehensive dictionary containing fitted parameters, global Z0 
        location analysis, complete distribution properties, optimization results, and diagnostic information for thorough global 
        distribution analysis.
    
        The QGDF results focus on global distribution properties and comprehensive critical point identification, particularly the 
        global Z0 point which represents the location of minimum probability density across the entire data range. This makes it 
        particularly valuable for system-wide reliability analysis, comprehensive critical point identification, enterprise quality 
        control threshold analysis, and understanding global distribution behavior that affects the complete data domain rather 
        than local regions only.
    
        The results include:
            - Precise global Z0 point location and identification method used
            - Global distribution bounds across complete data range (DLB, DUB, LB, UB)
            - Optimal scale parameter (S_opt) for global distribution fitting
            - QGDF values representing complete global distribution characteristics
            - PDF values with focus on minimum regions and global Z0 significance
            - Global Z0-focused evaluation points and comprehensive distribution analysis
            - Advanced global Z0 identification diagnostics and method validation
            - Optimization convergence information specific to global fitting
            - Error and warning logs from both global distribution fitting and Z0 identification
    
        Returns:
            dict: A comprehensive dictionary containing fitted parameters and global distribution results.
                  Primary keys include:
                  
                  Global Z0 Point Analysis:
                  - 'z0': Precisely identified global Z0 point location (absolute PDF minimum)
                  - 'z0_method': Method used for global Z0 identification (e.g., 'global_spline_optimization', 'global_polynomial_fitting')
                  - 'z0_pdf_value': PDF value at the global Z0 point (should be absolute minimum)
                  - 'z0_global_significance': Analysis of Z0 significance across entire data range
                  - 'z0_optimization_info': Detailed information about global Z0 identification process
                  - 'z0_convergence': Convergence information for global Z0 optimization
                  - 'z0_bounds_validation': Whether global Z0 location satisfies specified bounds
                  
                  Global Distribution Parameters:
                  - 'DLB': Data Lower Bound used for global distribution fitting
                  - 'DUB': Data Upper Bound used for global distribution fitting  
                  - 'LB': Lower Probable Bound for global distribution analysis
                  - 'UB': Upper Probable Bound for global distribution analysis
                  - 'S_opt': Optimal scale parameter estimated for global distribution
                  - 'global_scale_analysis': Analysis of scale parameter across entire range
                  
                  Global Distribution Functions:
                  - 'qgdf': QGDF values representing complete global distribution characteristics
                  - 'pdf': PDF values with comprehensive analysis across entire data range
                  - 'cdf': Cumulative distribution function values across complete domain
                  - 'qgdf_points': Points optimized for global distribution evaluation
                  - 'pdf_points': Points with comprehensive resolution across entire range
                  - 'zi': Transformed data points in global distribution domain
                  - 'zi_points': Corresponding evaluation points for global analysis
                  
                  Data and Processing Information:
                  - 'data': Original sorted input data used for global fitting
                  - 'weights': Weights applied to data points during global analysis
                  - 'wedf': Boolean indicating WEDF usage (typically False for QGDF)
                  - 'data_form': Data processing form ('a' additive, 'm' multiplicative)
                  - 'n_points': Number of points used for global distribution evaluation
                  - 'homogeneous': Data homogeneity assumption used in global fitting
                  
                  Optimization and Quality Metrics:
                  - 'fitted': Boolean confirming successful global distribution fitting
                  - 'z0_optimize': Boolean confirming global Z0 optimization was enabled
                  - 'tolerance': Convergence tolerance achieved in global optimization
                  - 'opt_method': Optimization method used for global parameter estimation
                  - 'max_data_size': Maximum data size limit applied during global processing
                  - 'flush': Whether memory flushing was used during global computation
                  
                  Advanced Global Diagnostics:
                  - 'warnings': List of warnings from global fitting and Z0 identification
                  - 'errors': List of errors encountered during global fitting (if any)
                  - 'optimization_history': Detailed record of global optimization iterations
                  - 'z0_identification_log': Step-by-step global Z0 identification process
                  - 'convergence_info': Information about global optimization convergence
                  - 'global_fit_quality': Metrics assessing global distribution fit quality
                  - 'z0_precision': Estimated precision of global Z0 identification
                  - 'z0_global_impact': Analysis of Z0 impact on entire distribution
                  - 'global_sensitivity': Sensitivity analysis of global parameters
    
        Raises:
            RuntimeError: If fit() has not been called before accessing results, or if global Z0 identification failed.
                         The QGDF model must be successfully fitted with global Z0 identification before results retrieval.
            AttributeError: If internal result structure is missing or corrupted due to global fitting failure.
            KeyError: If expected result keys are unavailable, possibly due to incomplete global fitting
                     or Z0 identification issues.
            ValueError: If internal state is inconsistent for result retrieval, which may occur
                       if the global distribution fitting or Z0 identification encountered issues.
            MemoryError: If results contain very large arrays that exceed available memory
                        (relevant for large datasets with high n_points values in global analysis).
    
        Side Effects:
            None. This method provides read-only access to global fitting results and does not modify
            the internal state of the QGDF object or trigger any recomputation of global Z0 identification.
    
        Examples:
            Basic usage after global distribution fitting with Z0 identification:
            >>> qgdf = QGDF(data, verbose=True, z0_optimize=True, tolerance=1e-9)
            >>> qgdf.fit()
            >>> results = qgdf.results()
            >>> print(f"Global Z0 point location: {results['z0']:.8f}")
            >>> print(f"Global Z0 identification method: {results['z0_method']}")
            >>> print(f"Global scale parameter: {results['S_opt']:.6f}")
            
            Comprehensive global Z0 analysis:
            >>> z0_location = results['z0']
            >>> z0_method = results['z0_method'] 
            >>> z0_pdf_value = results['z0_pdf_value']
            >>> global_bounds = (results['LB'], results['UB'])
            >>> print(f"Global Z0 at {z0_location:.8f} (PDF={z0_pdf_value:.8f}) via {z0_method}")
            
            Quality assessment of global Z0 identification:
            >>> if 'z0_convergence' in results:
            ...     conv_info = results['z0_convergence']
            ...     print(f"Global Z0 optimization converged: {conv_info.get('success', 'Unknown')}")
            >>> if results.get('z0_bounds_validation', False):
            ...     print("Global Z0 location validated within specified bounds")
            >>> if results['warnings']:
            ...     print(f"Global Z0 identification warnings: {len(results['warnings'])}")
            
            Advanced global Z0 diagnostics:
            >>> if 'z0_identification_log' in results:
            ...     log = results['z0_identification_log']
            ...     print(f"Global Z0 identification attempts: {len(log)}")
            ...     for entry in log:
            ...         method = entry.get('method', 'unknown')
            ...         success = entry.get('success', False)
            ...         print(f"  {method}: {'Success' if success else 'Failed'}")
            
            Global distribution analysis:
            >>> global_pdf = results['pdf']
            >>> z0_index = np.argmin(global_pdf)  # Should correspond to global Z0 location
            >>> zi_points = results['zi_points']
            >>> print(f"Global Z0 verification: PDF minimum at zi={zi_points[z0_index]:.8f}")
            
            Global optimization performance analysis:
            >>> if results.get('optimization_history'):
            ...     history = results['optimization_history']
            ...     print(f"Global optimization iterations: {len(history)}")
            ...     if history:
            ...         final_loss = history[-1].get('total_loss', 'N/A')
            ...         print(f"Final global optimization loss: {final_loss}")
    
        Applications of Global QGDF Results:
            Global Z0 Point Applications:
            - System-wide reliability engineering: Global critical failure points and stress thresholds
            - Enterprise quality control: Global specification boundaries and acceptable limit determination
            - Market-wide risk assessment: Global equilibrium points and maximum uncertainty analysis
            - Environmental policy: Global threshold detection and critical limit analysis for regulations
            - Supply chain optimization: Global bottleneck identification and critical point analysis
            - Network performance: Global system stress points and capacity limit determination
            - Scientific research: Global phase transitions and critical phenomena analysis
            
            Global Distribution Applications:
            - Comprehensive analysis across entire data range (complete system view)
            - Global probability density characterization for system-wide risk assessment
            - Enterprise-wide decision making using global Z0 location and distribution properties
            - Regulatory compliance with global threshold and specification analysis
            - Strategic planning with global distribution insights and critical point identification
    
        Interpretation Guide:
            Global Z0 Point Results:
            - 'z0': The precise location where PDF reaches absolute minimum across entire range
            - 'z0_method': Indicates reliability of global identification ('global_spline_optimization' > 'discrete_extremum')
            - 'z0_pdf_value': Should be the smallest value in the PDF array across entire range
            - Low 'z0_pdf_value' indicates well-defined global minimum region
            
            Global Distribution Parameters:
            - 'S_opt': Scale parameter controls global distribution spread across entire range
            - Bounds (LB, UB): Define effective global analysis range for complete distribution
            - Global scale analysis shows parameter behavior across complete data domain
            
            Quality Indicators:
            - Empty 'warnings' and 'errors' indicate successful global Z0 identification
            - 'z0_convergence.success=True' confirms reliable global Z0 optimization
            - 'z0_bounds_validation=True' indicates global Z0 within reasonable constraints
            - Advanced 'z0_method' (not 'discrete_extremum') suggests high global precision
    
        Performance Considerations:
            - Results retrieval is immediate and optimized for global Z0 analysis access
            - Global Z0 identification adds comprehensive diagnostic information to standard fitting results
            - Large n_points values increase global Z0 precision but also result array sizes
            - Global distribution evaluation provides comprehensive analysis across entire data range
            - Advanced global Z0 methods generate more detailed diagnostic information
            - Memory usage scales with both data size and global Z0 analysis detail level
    
        Comparison with Other GDF Results:
            QGDF vs QLDF Results:
            - QGDF: Global Z0 point (absolute PDF minimum) identification with comprehensive global analysis
            - QLDF: Local Z0 point (global PDF minimum) identification with local analysis around critical points
            
            QGDF vs EGDF/ELDF Results:
            - QGDF: Global distribution focus with detailed global Z0 critical point analysis
            - EGDF/ELDF: Global/local distribution analysis with focus on PDF maxima rather than minima
            
            QGDF Unique Features:
            - 'z0_method' indicates advanced global Z0 identification technique used
            - 'z0_identification_log' provides step-by-step global Z0 optimization history
            - 'z0_global_significance' and 'z0_global_impact' offer comprehensive global analysis
            - Global distribution focus providing complete system perspective rather than local features
    
        Notes:
            - fit() with z0_optimize=True must complete successfully before accessing global Z0 results
            - Global Z0 identification may generate extensive diagnostic information for complex data
            - Results structure is consistent regardless of global Z0 identification method succeeded
            - Advanced global Z0 methods (spline, polynomial) provide more detailed diagnostics
            - Global distribution focus means results emphasize complete range characteristics
            - If catch=False was used, some detailed global Z0 intermediate results may be unavailable
            - Global Z0 location represents absolute PDF minimum across entire data range
            - All numeric results use high precision appropriate for global Z0 analysis
            - Results can be serialized for reproducible global Z0 analysis and reporting
    
        Troubleshooting:
            Incomplete Global Z0 Results:
            - Verify fit() completed without exceptions and global Z0 optimization was enabled
            - Check 'z0_identification_log' for specific global Z0 identification failures
            - Ensure adequate tolerance for global Z0 optimization convergence
            
            Unexpected Global Z0 Location:
            - Cross-reference 'z0' with PDF values to confirm it's at absolute minimum
            - Check 'z0_bounds_validation' for global constraint satisfaction
            - Review 'z0_method' - advanced methods are more reliable for global analysis
            - Examine 'z0_global_impact' if available for Z0 significance across entire range
            
            Poor Global Z0 Identification Quality:
            - Check 'warnings' for insights into global Z0 identification challenges
            - Verify data has sufficient global variation for meaningful Z0 identification
            - Consider tighter tolerance or different optimization method for global Z0
            - Ensure global bounds appropriately constrain Z0 search across entire range
            
            Global Z0 Optimization Issues:
            - Review 'z0_convergence' information for global optimization problems
            - Check 'z0_identification_log' for method-specific failure patterns
            - Consider enabling verbose=True during fitting for global Z0 diagnostic output
            - Try different global data preprocessing or bound specifications for Z0 search
        """
        if not self._fitted:
            raise RuntimeError("Must fit QGDF before getting results.")
        
        return self._get_results()