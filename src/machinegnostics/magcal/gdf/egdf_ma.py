'''
EGDF Marginal Analysis Module

Machine Gnostics
Author: Nirmal Parmar
'''

import numpy as np
from machinegnostics.magcal.gdf.egdf import EGDF
from machinegnostics.magcal.gdf.base_eg_ma import BaseMarginalAnalysisEGDF

class MarginalAnalysisEGDF(BaseMarginalAnalysisEGDF):
    """
    Marginal Analysis class for Extended GDF (EGDF) with advanced boundary estimation capabilities.
    
    This class performs comprehensive marginal analysis on data samples to identify critical boundaries
    and intervals that characterize the underlying data distribution. It extends the base EGDF functionality
    with specialized algorithms for boundary detection, clustering, and interval analysis.
    
    ### Key Features:
    
    **Marginal Analysis - Boundary Estimation:**
    
    1. **LB and UB (Lower/Upper Bounds)**: Probable bounds of the data sample optimized during EGDF fitting.
       These represent the statistical boundaries where the distribution function transitions.
    
    2. **LSB and USB (Lower/Upper Sample Bounds)**: Extreme limits for data homogeneity. These are the
       outermost boundaries where the data maintains statistical consistency with the fitted distribution.
    
    3. **DLB and DUB (Data Lower/Upper Bounds)**: Actual minimum and maximum values in the data sample.
       These represent the observed range of the data.
    
    4. **CLB and CUB (Cluster Lower/Upper Bounds)**: Boundaries identifying the main data clusters.
       Used for detecting significant data groupings and distribution modes.

    5. **Z0**: Central point where PDF reaches global maximum and EGDF ≈ 0.5. This represents the
       distribution's central tendency and inflection point.
    
    ### Use Cases:
    
    - **Quality Control**: Identifying acceptable data ranges and outlier boundaries
    - **Statistical Modeling**: Understanding data distribution characteristics
    - **Anomaly Detection**: Setting thresholds for unusual data points
    - **Process Monitoring**: Establishing control limits for manufacturing processes
    - **Risk Assessment**: Defining confidence intervals for decision making

    ### Attributes:

    data : np.ndarray
        Input data array (1-dimensional) for marginal analysis. Must be a 1D numpy array
        containing numerical values. Empty arrays or arrays with all NaN values
        will raise an error.
        
    sample_bound_tolerance : float, default=0.1
        Tolerance level for sample bound estimation (LSB/USB detection). Controls the precision
        of boundary optimization. Smaller values provide more precise bounds but require more 
        iterations. This is the convergence tolerance for derivative-based optimization.
        
    max_iterations : int, default=10000
        Maximum number of iterations for boundary optimization algorithms. Higher values
        allow more thorough optimization but increase computation time. Must be positive integer.
        
    early_stopping_steps : int, default=10
        Number of consecutive steps without improvement before stopping optimization.
        Prevents infinite loops and improves efficiency. Must be positive integer.
        
    estimating_rate : float, default=0.1
        Learning rate for gradient-based boundary estimation. Controls convergence speed
        and stability. Smaller values provide more stable convergence but slower optimization.
        Must be positive value typically between 0.01 and 1.0.
        
    cluster_threshold : float, default=0.05
        Threshold for PDF-based cluster detection as fraction of maximum PDF value.
        Lower values detect more subtle clusters. Range typically 0.01 to 0.2.
        
    get_clusters : bool, default=True
        Whether to perform cluster analysis and compute CLB/CUB bounds. When True,
        enables cluster boundary detection. Set to False to skip clustering for faster processing.
        
    DLB : float, optional
        Data Lower Bound - the absolute minimum value that the data can theoretically take.
        If None, will be inferred from data minimum. This is a hard constraint on the distribution.
        Manual override for data lower bound.
        
    DUB : float, optional
        Data Upper Bound - the absolute maximum value that the data can theoretically take.
        If None, will be inferred from data maximum. This is a hard constraint on the distribution.
        Manual override for data upper bound.
        
    LB : float, optional
        Lower Probable Bound - the practical lower limit for the distribution.
        This is typically less restrictive than DLB and represents the expected
        lower range of the distribution. Manual override for EGDF lower bound.
        
    UB : float, optional
        Upper Probable Bound - the practical upper limit for the distribution.
        This is typically less restrictive than DUB and represents the expected
        upper range of the distribution. Manual override for EGDF upper bound.
        
    S : float or 'auto', default='auto'
        Scale parameter for the distribution. If 'auto' (default), the scale will be 
        automatically estimated from the data during fitting. If a float is provided, 
        it will be used as a fixed scale parameter. Affects distribution spread.
        
    tolerance : float, default=1e-6
        Numerical tolerance for convergence criteria in optimization algorithms.
        Smaller values lead to more precise fitting but may require more iterations.
        Controls precision of all numerical computations.
        
    data_form : str, default='a'
        Form of data processing. Options are:
        - 'a': Additive form (default) - processes data linearly
        - 'm': Multiplicative form - applies log transformation for
               better handling of multiplicative processes
        
    n_points : int, default=1000
        Number of points to generate in the final distribution function for smooth
        curve generation in plotting and analysis. Higher values provide smoother 
        curves but require more computation. Must be positive integer.
        
    homogeneous : bool, default=True
        Whether to assume data homogeneity. Affects boundary estimation algorithms
        and internal optimization strategies. Set to False for heterogeneous data.
        
    catch : bool, default=True
        Whether to enable error catching and provide detailed analysis results.
        Setting to True (default) allows access to detailed results and plotting
        but uses more memory. Required for plotting and parameter access.
        
    weights : np.ndarray, optional
        Sample weights for weighted EGDF analysis. Must be the same length as data array.
        If None, uniform weights (all ones) are used. Weights should be positive values.
        Prior weights for data points to emphasize certain observations.
        
    wedf : bool, default=True
        Whether to compute Weighted Empirical Distribution Function (WEDF) alongside 
        standard EGDF. When True, incorporates weights into empirical distribution estimation.
        When False, uses Kolmogorov-Smirnov (KS) Points for distribution analysis.
        
    opt_method : str, default='L-BFGS-B'
        Optimization method for EGDF parameter estimation. Default is 'L-BFGS-B'.
        Other options include 'TNC', 'Powell', 'SLSQP', etc. Must be a valid 
        scipy.optimize method name that supports bounds.
        
    verbose : bool, default=False
        Whether to print detailed progress information during fitting. When True,
        provides diagnostic output about the optimization process and boundary detection.
        
    max_data_size : int, default=1000
        Maximum data size for processing. Safety limit to prevent excessive memory usage
        during boundary estimation and smooth curve generation.
        
    flush : bool, default=True
        Whether to flush output streams for real-time progress display and intermediate
        calculations during processing. May affect memory usage and computation speed.
        Default is True for better user feedback.
    
    
    ### Examples

    Basic usage for quality control analysis:
    
    >>> import numpy as np
    >>> from machinegnostics.magcal import MarginalAnalysisEGDF
    >>> 
    >>> # Manufacturing measurement data
    >>> data = np.array([-10,-9,-8,-0.2,-0.1,0,0.1,0.2,8,9,10] )
    >>> 
    >>> # Perform marginal analysis
    >>> ma = MarginalAnalysisEGDF(data=data, verbose=True)
    >>> ma.fit()
    >>> 
    >>> # Access computed boundaries
    >>> print(f"Data range: [{ma.DLB:.3f}, {ma.DUB:.3f}]")
    >>> print(f"Sample bounds: [{ma.LSB:.3f}, {ma.USB:.3f}]")
    >>> print(f"Central point Z0: {ma.z0:.3f}")
    
    ### Notes

    - The algorithm uses iterative optimization with early stopping for efficiency
    - Boundary estimation accuracy depends on data quality and sample size
    - CLB/CUB bounds are only computed when get_clusters=True
    - Setting catch=False disables smooth plotting and detailed parameter access
    - Bounds should be chosen carefully: too restrictive bounds may lead to poor fits
    - For multiplicative data, consider using data_form='m' for better results
    - Large n_points values will slow down plotting but provide smoother visualizations
    
    ### Raises

    ValueError
        If data array is empty, contains only NaN values, or has invalid dimensions.
        If weights array is provided but has different length than data array.
        If bounds are specified incorrectly (e.g., DLB > DUB or LB > UB).
        If tolerance, max_iterations, or other numerical parameters are invalid.
        
    ### RuntimeError
        If EGDF fitting fails or optimization fails to converge.
        If boundary estimation cannot find valid LSB/USB values.

    ### OptimizationError
        If the underlying optimization algorithm encounters numerical issues.
    
    """

    def __init__(self,
                data: np.ndarray,
                sample_bound_tolerance: float = 0.1,
                max_iterations: int = 10000,
                early_stopping_steps: int = 10,
                estimating_rate: float = 0.1,
                cluster_threshold: float = 0.05,
                get_clusters: bool = True,
                DLB: float = None,
                DUB: float = None,
                LB: float = None,
                UB: float = None,
                S = 'auto',
                tolerance: float = 1e-6,
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


    def fit(self, plot=False):
        """
        Fit the EGDF Marginal Analysis model and perform comprehensive marginal analysis.
        
        This method executes the complete marginal analysis workflow:
        1. Fits the initial EGDF to the data, estimate [DLB, DUB] and [LB, UB] bounds
        2. Estimates sample bounds (LSB, USB) using derivative optimization
        3. Identifies the central point Z0 where PDF reaches maximum
        4. Performs cluster analysis to find CLB and CUB bounds
        5. Tests data homogeneity
        6. Stores all results in the params dictionary
        
        The fitting process uses advanced numerical optimization techniques:
        - Gradient descent with adaptive learning rates for boundary estimation
        - Newton-Raphson method for Z0 detection
        - PDF-based clustering with gradient analysis
        - Early stopping
        
        Parameters
        ----------
        plot : bool, default=True
            Whether to display a summary plot after fitting.
            If True, shows EGDF, PDF, and all identified boundaries.
            Requires catch=True to be set during initialization.
        
        compute_z0 : bool, default=True
            Whether to compute the central point Z0 during fitting.
            If True, estimates the point where PDF reaches maximum
        
        Returns
        -------
        None
            Results are stored in instance attributes:
            - self.params: Dictionary with all computed parameters
            - self.LSB, self.USB: Sample bound values
            - self.init_egdf: Fitted EGDF object
        
        Raises
        ------
        RuntimeError
            If EGDF fitting fails or data is invalid.
            
        ValueError
            If optimization fails to converge within max_iterations.
            
        Examples
        --------
        Basic fitting:
        
        >>> ma = MarginalAnalysisEGDF(data=np.array([1, 2, 3, 4, 5]))
        >>> ma.fit()  # Fits model and shows plot
        
        Fitting without plotting:
        
        >>> ma.fit(plot=False)  # Silent fitting
        >>> print(f"Z0 = {ma.z0:.3f}")
        
        High-precision fitting:
        
        >>> ma = MarginalAnalysisEGDF(
        ...     data=data, 
        ...     sample_bound_tolerance=0.001,
        ...     max_iterations=20000
        ... )
        >>> ma.fit(plot=True)
        
        Notes
        -----
        - Fitting time depends on data size and tolerance settings
        - For large datasets, consider reducing n_points for faster fitting
        - If convergence issues occur, try increasing max_iterations or tolerance
        - The plot parameter is ignored if catch=False was set during initialization
        """
        self._fit_egdf(plot=plot)

    def plot(self, plot_type='marginal', plot_smooth=True, bounds=True, figsize=(12, 8)):
        """
        Generate comprehensive visualization of EGDF marginal analysis results.
        
        Creates publication-quality plots showing the fitted EGDF, PDF, and all identified
        boundaries and critical points. Supports multiple plot types and customization options
        for different analysis needs.
        
        The plotting system uses dual y-axes to clearly show both cumulative (EGDF) and 
        density (PDF) functions, with color-coded boundary markers and legends for easy
        interpretation.
        
        Parameters
        ----------
        plot_type : {'marginal', 'egdf', 'pdf', 'both'}, default='marginal'
            Type of plot to generate:
            
            - 'marginal': Shows both EGDF and PDF with all boundaries (recommended)
            - 'egdf': Shows only EGDF with boundaries
            - 'pdf': Shows only PDF with boundaries  
            - 'both': Alias for 'marginal'
            
        plot_smooth : bool, default=True
            Whether to plot smooth interpolated curves:
            
            - True: Shows both discrete points and smooth curves for publication quality
            - False: Shows only discrete data points for detailed analysis
            
        bounds : bool, default=True
            Controls which boundaries are displayed:
            
            - True: Shows all bounds (DLB, DUB, LB, UB, LSB, USB, CLB, CUB) plus Z0
            - False: Shows only Z0 (central point)
            
        figsize : tuple, default=(12, 8)
            Figure size as (width, height) in inches.
            Adjust for different display requirements or publication formats.
        
        Returns
        -------
        None
            Displays the plot using matplotlib. No return value.
        
        Raises
        ------
        RuntimeError
            If fit() has not been called yet or if catch=False was set during initialization.
            
        ValueError
            If plot_type is not one of the supported options.
        
        Examples
        --------
        Standard marginal analysis plot:
        
        >>> ma = MarginalAnalysisEGDF(data=data)
        >>> ma.fit()
        >>> ma.plot()  # Shows everything with default settings
        
        Customized visualization:
        
        >>> # Large figure for presentation
        >>> ma.plot(
        ...     plot_type='marginal',
        ...     plot_smooth=True, 
        ...     bounds=True,
        ...     figsize=(16, 10)
        ... )
                
        Plot Elements
        -------------
        **Lines and Curves:**
        - Blue solid line: EGDF (cumulative distribution)
        - Light blue squares: WEDF if computed
        - Red solid line: PDF (probability density)
        - Smooth curves: Interpolated versions when plot_smooth=True
        
        **Boundary Markers (when bounds=True):**
        - Green solid: DLB (Data Lower Bound)
        - Orange solid: DUB (Data Upper Bound)  
        - Purple dashed: LB (EGDF Lower Bound)
        - Brown dashed: UB (EGDF Upper Bound)
        - Dark red dotted: LSB (Lower Sample Bound)
        - Dark blue dotted: USB (Upper Sample Bound)
        - Orange dashed: CLB, CUB (Cluster Bounds)
        - Magenta dash-dot: Z0 (Central Point) - always shown
        
        **Shaded Regions:**
        - Light purple: Region below LB
        - Light brown: Region above UB
        
        **Axes:**
        - Primary y-axis (left): EGDF values [0, 1]
        - Secondary y-axis (right): PDF values [0, max_pdf]
        - X-axis: Data range with 5% padding
        
        Notes
        -----
        - Requires catch=True during initialization for plotting capability
        - Plot automatically adjusts scales and ranges based on data characteristics  
        - Legend shows all visible elements with their numerical values
        - Grid is enabled with low opacity for better readability
        - Color scheme is optimized for both screen display and printing
        
        See Also
        --------
        fit : Must be called before plotting
        """
        self._plot_egdf(plot_type=plot_type, plot_smooth=plot_smooth, bounds=bounds, figsize=figsize)