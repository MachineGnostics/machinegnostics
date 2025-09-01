'''
Public class EGDf Interval Analysis
This class work on top of BaseIntervalAnalysisEGDF and BaseMarginalAnalysisEGDF.

EGDF class is imported in BaseMarginalAnalysisEGDF

Machine Gnostics
Author: Nirmal Parmar
'''

import numpy as np
from pyparsing import Any
from machinegnostics.magcal.gdf.base_eg_intv import BaseIntervalAnalysisEGDF

class IntervalAnalysisEGDF(BaseIntervalAnalysisEGDF):
    """
    Interval Analysis class for Extended GDF (EGDF) with advanced interval detection capabilities.
    
    This class performs comprehensive interval analysis on data samples to identify critical intervals
    and boundaries that characterize the underlying data distribution. It extends the base EGDF functionality
    with specialized algorithms for interval detection and boundary analysis.
    
    ### Key Features:
    
    **Interval Analysis - Interval Detection:**
    
    1. **Tolerance Interval (Z0L, Z0U)**: 
       A finite interval that defines the location parameter's tolerance under all possible 
       changes in the data. The location parameter Z̃0 cannot leave this interval 
       for given data range. 
       This interval demonstrates the robustness of the EGDF's location parameter and is always smaller than the typical data interval.
    
    2. **Typical Data Interval (ZL, ZU)**:
       The interval where data behavior can be considered "typical" or normal. Within this 
       interval, increasing the extending data item Z leads to increasing the location 
       parameter Z̃0. Data points within this interval follow the expected pattern of the underlying distribution.
    
    3. **Atypical Data Intervals**:
       - **Lower Atypical (LB, ZL)**: Data region below the typical interval where behavior deviates from normal patterns
       - **Upper Atypical (ZU, UB)**: Data region above the typical interval where behavior deviates from normal patterns

    ### Use Cases:
    
    - **Data Segmentation**: Partitioning continuous data into meaningful intervals
    - **Anomaly Detection**: Identifying unusual data regions through interval analysis
    - **Quality Control**: Establishing interval-based control limits for processes
    - **Pattern Recognition**: Detecting recurring patterns in time series or sequential data
    - **Feature Engineering**: Creating interval-based features for machine learning
    - **Risk Management**: Defining risk intervals for financial or operational data

    ### Attributes:

    data : np.ndarray
        Input data array (1-dimensional) for interval analysis. Must be a 1D numpy array
        containing numerical values. Empty arrays or arrays with all NaN values
        will raise an error.
        
    estimate_sample_bounds : bool, default=False
        Whether to estimate sample bounds (LSB/USB) during interval analysis. When True,
        computes extreme boundaries for data homogeneity analysis.
        
    estimate_cluster_bounds : bool, default=False
        Whether to estimate cluster bounds (CLB/CUB) during interval analysis. When True,
        performs clustering analysis to identify main data groupings by removing the outliers.

    sample_bound_tolerance : float, default=0.1
        Tolerance level for sample bound estimation. Controls the precision of boundary
        optimization when estimate_sample_bounds=True. Smaller values provide more precise 
        bounds but require more iterations.
        
    max_iterations : int, default=1000
        Maximum number of iterations for interval optimization algorithms. Higher values
        allow more thorough optimization but increase computation time. Must be positive integer.
        Specific to interval analysis optimization processes.
        
    early_stopping_steps : int, default=10
        Number of consecutive steps without improvement before stopping optimization.
        Prevents infinite loops and improves efficiency during interval detection.
        
    estimating_rate : float, default=0.1
        Learning rate for gradient-based interval boundary estimation. Controls convergence speed
        and stability during optimization. Smaller values provide more stable convergence.
        Must be positive value typically between 0.01 and 1.0. Specific to interval analysis.
        
    cluster_threshold : float, default=0.05
        Threshold for PDF-based cluster detection as fraction of maximum PDF value.
        Lower values detect more subtle clusters and finer intervals. Range typically 0.01 to 0.2.
    
    linear_search : bool, default=True
        Whether to use linear search for interval boundary detection. When True, employs a
        straightforward linear search algorithm for simplicity and interpretability. When False,
        uses more complex SciPy optimization methods for potentially faster convergence.

    get_clusters : bool, default=False
        Whether to perform cluster analysis during interval detection. When True,
        enables cluster-based interval identification. Set to False for faster processing
        without clustering. Specific to interval analysis workflow.
        
    DLB : float, optional
        Data Lower Bound - the absolute minimum value that the data can theoretically take.
        If None, will be inferred from data minimum. Manual override for distribution lower bound.
        
    DUB : float, optional
        Data Upper Bound - the absolute maximum value that the data can theoretically take.
        If None, will be inferred from data maximum. Manual override for distribution upper bound.
        
    LB : float, optional
        Lower Probable Bound - the practical lower limit for interval analysis.
        Manual override for EGDF lower bound used in interval computations.
        
    UB : float, optional
        Upper Probable Bound - the practical upper limit for interval analysis.
        Manual override for EGDF upper bound used in interval computations.
        
    S : float or 'auto', default='auto'
        Scale parameter for the distribution. If 'auto' (default), the scale will be 
        automatically estimated from the data during fitting. Affects interval detection sensitivity.
        
    tolerance : float, default=1e-6
        Numerical tolerance for convergence criteria in interval optimization algorithms.
        Smaller values lead to more precise interval boundaries but may require more iterations.
        Specific to interval analysis numerical computations.
        
    data_form : str, default='a'
        Form of data processing for interval analysis. Options are:
        - 'a': Additive form (default) - processes intervals linearly
        - 'm': Multiplicative form - applies log transformation for better handling
               of multiplicative processes in interval detection
        
    n_points : int, default=1000
        Number of points to generate for interval analysis and smooth curve generation.
        Higher values provide more precise interval boundaries and smoother visualizations
        but require more computation. Must be positive integer. Specific to interval analysis.
        
    homogeneous : bool, default=True
        Whether to assume data homogeneity during interval analysis. Affects interval
        boundary estimation algorithms and optimization strategies.
        
    catch : bool, default=True
        Whether to enable error catching and provide detailed interval analysis results.
        Setting to True (default) allows access to detailed results and interval plotting
        but uses more memory. Required for interval plotting and parameter access.
        
    weights : np.ndarray, optional
        Sample weights for weighted interval analysis. Must be the same length as data array.
        If None, uniform weights are used. Affects interval boundary detection priorities.
        
    wedf : bool, default=True
        Whether to compute Weighted Empirical Distribution Function (WEDF) for interval analysis.
        When True, incorporates weights into interval detection algorithms.
        
    opt_method : str, default='L-BFGS-B'
        Optimization method for interval boundary estimation. Default is 'L-BFGS-B'.
        Must be a valid scipy.optimize method that supports bounds constraints.
        
    verbose : bool, default=False
        Whether to print detailed progress information during interval analysis.
        When True, provides diagnostic output about optimization and interval detection.
        
    max_data_size : int, default=1000
        Maximum data size for interval processing. Safety limit to prevent excessive memory usage
        during interval boundary estimation and smooth curve generation.
        
    flush : bool, default=True
        Whether to flush output streams for real-time progress display during interval processing.
        May affect memory usage and computation speed during interval analysis.
    
    ### Examples

    Basic interval analysis:
    
    >>> import numpy as np
    >>> from machinegnostics.magcal import IntervalAnalysisEGDF
    >>> 
    >>> # Sample data with multiple intervals
    >>> data = np.array([-10, -9, -8, -0.2, -0.1, 0, 0.1, 0.2, 8, 9, 10])
    >>> 
    >>> # Perform interval analysis
    >>> ia = IntervalAnalysisEGDF(
    ...     data=data, 
    ...     max_iterations=1000,
    ...     tolerance=1e-6,
    ...     verbose=True
    ... )
    >>> ia.fit()
    >>> 
    >>> # Get detected intervals
    >>> intervals = ia.get_intervals(decimals=3)
    >>> print("Detected intervals:", intervals)
    >>> 
    >>> # Plot interval analysis results and bounds
    >>> ia.plot()
    
    ### Methods

    fit(plot=True)
        Fit the EGDF Interval Analysis model to the data.

    get_intervals(decimals=2)
        Return dictionary containing all detected intervals with specified precision.
        
    plot(plot_type='marginal', plot_smooth=True, bounds=True, intervals=True, ...)
        Plot the EGDF analysis results with interval visualization.
        
    plot_intervals(plot_style='scatter', show_data_points=True, ...)
        Specialized plot focusing on interval visualization of the interval analysis.
    
    ### Notes

    - Interval analysis is computationally more intensive than marginal analysis
    - The algorithm uses iterative optimization with early stopping for efficiency
    - Interval detection accuracy depends on data quality, sample size, and tolerance settings
    - Setting get_clusters=True enables more sophisticated interval detection
    - For large datasets, consider reducing n_points for faster processing
    - The tolerance parameter may affect interval boundary precision

    ### Raises

    ValueError
        If data array is empty, contains only NaN values, or has invalid dimensions.
        If weights array is provided but has different length than data array.
        If numerical parameters (tolerance, max_iterations, etc.) are invalid.
        
    RuntimeError
        If EGDF fitting fails or interval optimization fails to converge.
        If interval detection cannot find valid boundaries within max_iterations.

    OptimizationError
        If the underlying optimization algorithm encounters numerical issues during
        interval boundary estimation.
    
    """

    def __init__(self,
                data: np.ndarray,
                estimate_sample_bounds: bool = False,
                estimate_cluster_bounds: bool = False,
                sample_bound_tolerance: float = 0.1,
                max_iterations: int = 1000, # NOTE for intv specific
                early_stopping_steps: int = 10,
                estimating_rate: float = 0.1, # NOTE for intv specific
                cluster_threshold: float = 0.05,
                linear_search: bool = True, # NOTE for intv specific
                get_clusters: bool = False, # NOTE for intv specific
                DLB: float = None,
                DUB: float = None,
                LB: float = None,
                UB: float = None,
                S = 'auto',
                tolerance: float = 1e-5, # NOTE for intv specific
                data_form: str = 'a',
                n_points: int = 1000, # NOTE for intv specific
                homogeneous: bool = True,
                catch: bool = True,
                weights: np.ndarray = None,
                wedf: bool = True,
                opt_method: str = 'L-BFGS-B',
                verbose: bool = False,
                max_data_size: int = 1000,
                flush: bool = True):
        super().__init__(
            data=data,
            estimate_sample_bounds=estimate_sample_bounds,
            estimate_cluster_bounds=estimate_cluster_bounds,
            sample_bound_tolerance=sample_bound_tolerance,
            max_iterations=max_iterations,
            early_stopping_steps=early_stopping_steps,
            estimating_rate=estimating_rate,
            cluster_threshold=cluster_threshold,
            linear_search=linear_search,
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
            flush=flush
        )

    def fit(self, plot=False):
        """
        Fit the EGDF Interval Analysis model to the data and detect critical intervals.
        
        This method performs the complete interval analysis workflow including:
        - Fitting the Extended Generalized Distribution Function (EGDF) to the data
        - Detecting tolerance intervals (Z0,L, Z0,U) that bound the location parameter
        - Identifying typical data intervals (ZL, ZU) where data behavior is normal
        - Finding atypical intervals (LB, ZL) and (ZU, UB) with deviant behavior
        - Optionally performing cluster-based interval detection if get_clusters=True
        - Computing sample bounds (LSB, USB) if estimate_sample_bounds=True
        - Estimating cluster bounds (CLB, CUB) if estimate_cluster_bounds=True
        
        The fitting process uses iterative optimization with early stopping to ensure
        robust interval detection while maintaining computational efficiency.
        
        Parameters
        ----------
        plot : bool, default=True
            Whether to automatically plot the results after fitting. When True,
            generates a comprehensive visualization showing the EGDF curve, detected
            intervals, and boundaries. Set to False for programmatic use without
            visualization.
        
        Returns
        -------
        None
            This method modifies the object in-place, storing all fitted parameters
            and detected intervals as instance attributes accessible through
            get_intervals() method.
        
        Raises
        ------
        ValueError
            If the data array is empty, contains only NaN values, or has invalid format.
            If any of the fitting parameters (tolerance, max_iterations, etc.) are invalid.
        
        RuntimeError
            If the EGDF fitting process fails to converge within max_iterations.
            If interval detection cannot find valid boundaries due to data issues.
        
        OptimizationError
            If the underlying optimization algorithm encounters numerical issues
            during interval boundary estimation.
        
        Examples
        --------
        >>> import numpy as np
        >>> from machinegnostics.magcal import IntervalAnalysisEGDF
        >>> 
        >>> # Fit with automatic plotting
        >>> data = np.array([1, 2, 3, 5, 8, 9, 15, 20, 25])
        >>> ia = IntervalAnalysisEGDF(data=data, verbose=True)
        >>> ia.fit(plot=True)
        >>> 
        >>> # Fit without plotting for programmatic use
        >>> ia.fit(plot=False)
        >>> intervals = ia.get_intervals()
        
        Notes
        -----
        - The tolerance interval is always finite and demonstrates parameter robustness
        - Typical intervals represent regions following expected gnostic patterns
        - Computational time increases with max_iterations and n_points parameters
        - Setting get_clusters=True enables more sophisticated interval detection
        - The method must be called before accessing intervals or plotting results
        """
        self._fit_egdf_intv(plot=plot)
    
    def get_intervals(self, decimals: int = 2) -> dict:
        """
        Retrieve all detected intervals from the fitted EGDF Interval Analysis model.
        
        This method returns a comprehensive dictionary containing all intervals identified
        during the fitting process, including tolerance intervals, typical data intervals,
        atypical intervals, and optionally cluster-based intervals and sample bounds.
        
        Parameters
        ----------
        decimals : int, default=2
            Number of decimal places to round the interval boundaries. Higher values
            provide more precision but may include numerical noise. Must be non-negative.
            Typical range is 0-6 depending on data scale and precision requirements.
        
        Returns
        -------
        dict
            Dictionary containing detected intervals with the following keys:
            
            - 'tolerance_interval' : tuple
                (Z0L, Z0U) - Finite interval bounding the location parameter
            - 'typical_interval' : tuple
                (ZL, ZU) - Interval where data behavior is considered normal
            - 'lower_atypical' : tuple
                (LB, ZL) - Lower atypical interval with deviant behavior
            - 'upper_atypical' : tuple
                (ZU, UB) - Upper atypical interval with deviant behavior
            - 'sample_bounds' : tuple, optional
                (LSB, USB) - Sample bounds if estimate_sample_bounds=True
            - 'cluster_bounds' : tuple, optional
                (CLB, CUB) - Cluster bounds if estimate_cluster_bounds=True
            - 'data_bounds' : tuple
                (DLB, DUB) - Theoretical data bounds
            - 'probable_bounds' : tuple
                (LB, UB) - Practical probable bounds
        
        Raises
        ------
        RuntimeError
            If the model has not been fitted yet. Call fit() method first.
        
        ValueError
            If decimals parameter is negative or not an integer.
        
        Examples
        --------
        >>> import numpy as np
        >>> from machinegnostics.magcal import IntervalAnalysisEGDF
        >>> 
        >>> # Basic interval retrieval
        >>> data = np.array([1, 2, 3, 5, 8, 9, 15, 20, 25])
        >>> ia = IntervalAnalysisEGDF(data=data)
        >>> ia.fit(plot=False)
        >>> intervals = ia.get_intervals(decimals=3)
        >>> 
        >>> # Access core interval values
        >>> print("Tolerance interval:", (intervals['Z0L'], intervals['Z0U']))
        >>> print("Typical interval:", (intervals['ZL'], intervals['ZU']))
        >>> print("Location parameter:", intervals['Z0'])
        >>> 
        >>> # Access boundary values
        >>> print("Data bounds:", (intervals['DLB'], intervals['DUB']))
        >>> print("Probable bounds:", (intervals['LB'], intervals['UB']))
        >>> print("Sample bounds:", (intervals['LSB'], intervals['USB']))
        >>> 
        >>> # Derive atypical intervals
        >>> lower_atypical = (intervals['LB'], intervals['ZL'])
        >>> upper_atypical = (intervals['ZU'], intervals['UB'])
        >>> print("Lower atypical interval:", lower_atypical)
        >>> print("Upper atypical interval:", upper_atypical)
        >>> 
        >>> # High precision for detailed analysis
        >>> precise_intervals = ia.get_intervals(decimals=6)
        >>> tolerance_width = precise_intervals['Z0U'] - precise_intervals['Z0L']
        >>> typical_width = precise_intervals['ZU'] - precise_intervals['ZL']
        >>> print(f"Tolerance interval width: {tolerance_width}")
        >>> print(f"Typical interval width: {typical_width}")
        
        Notes
        -----
        - The tolerance interval demonstrates the robustness of the location parameter
        - Typical intervals indicate regions where increasing data leads to increasing location parameter
        - Atypical intervals show regions with deviant gnostic behavior
        - All intervals are returned as tuples of (lower_bound, upper_bound)
        - Missing intervals (e.g., when bounds estimation is disabled) return None
        - Interval precision depends on the tolerance parameter used during fitting
        """
        return self._get_intv(decimals=decimals)

    def plot(self, plot_type: str = 'marginal',
                plot_smooth: bool = True,
                bounds: bool = True,
                intervals: bool = True,
                show_all_bounds: bool = False,
                figsize: tuple = (12, 8)):
        """
        Plot comprehensive EGDF Interval Analysis results with customizable visualization options.
        
        This method generates detailed plots showing the fitted EGDF curve, detected intervals,
        boundaries, and data distribution. It provides multiple plot types and customization
        options for thorough analysis and presentation of interval detection results.
        
        Parameters
        ----------
        plot_type : str, default='marginal'
            Type of plot to generate. Options include:
            - 'marginal': Shows EGDF curve with marginal analysis features
            - 'interval': Focuses on interval visualization and boundaries
            - 'combined': Shows both marginal and interval analysis results
            - 'distribution': Emphasizes underlying data distribution
        
        plot_smooth : bool, default=True
            Whether to plot smoothed EGDF curves for better visualization.
            When True, uses n_points parameter to generate smooth curves.
            Set to False for faster plotting with original data points only.
        
        bounds : bool, default=True
            Whether to display boundary lines on the plot including:
            - Data bounds (DLB, DUB)
            - Probable bounds (LB, UB)  
            - Sample bounds (LSB, USB) if estimated
            - Cluster bounds (CLB, CUB) if estimated
        
        intervals : bool, default=True
            Whether to highlight detected intervals on the plot:
            - Tolerance interval with distinct shading
            - Typical interval with normal region highlighting
            - Atypical intervals with deviation region marking
        
        show_all_bounds : bool, default=False
            Whether to display all available bounds including optional ones.
            When True, shows sample bounds and cluster bounds even if they
            overlap with other boundaries. Useful for detailed analysis.
        
        figsize : tuple, default=(12, 8)
            Figure size as (width, height) in inches. Larger sizes provide
            more detail but consume more memory. Adjust based on display
            requirements and available screen space.
        
        Returns
        -------
        None
            Displays the plot using matplotlib. The plot is rendered immediately
            and can be saved using standard matplotlib commands if needed.
        
        Raises
        ------
        RuntimeError
            If the model has not been fitted yet. Call fit() method first.
        
        ValueError
            If plot_type is not one of the supported options.
            If figsize contains non-positive values.
        
        Examples
        --------
        >>> import numpy as np
        >>> from machinegnostics.magcal import IntervalAnalysisEGDF
        >>> 
        >>> # Basic plotting with all features
        >>> data = np.array([1, 2, 3, 5, 8, 9, 15, 20, 25])
        >>> ia = IntervalAnalysisEGDF(data=data)
        >>> ia.fit(plot=False)
        >>> ia.plot(plot_type='marginal', intervals=True, bounds=True)
        >>> 
        >>> # Focused interval visualization
        >>> ia.plot(plot_type='interval', show_all_bounds=True, figsize=(15, 10))
        >>> 
        >>> # Minimal plot for presentations
        >>> ia.plot(plot_smooth=True, bounds=False, intervals=True, figsize=(10, 6))
        
        Notes
        -----
        - The plot automatically adjusts scales and labels based on data range
        - Tolerance intervals are highlighted with distinctive colors/patterns
        - Typical and atypical regions are clearly distinguished visually
        - Boundary lines help assess interval positioning relative to data limits
        - Smooth curves provide better visual understanding of EGDF behavior
        - Large datasets may benefit from plot_smooth=False for faster rendering
        """
        self._plot_egdf(
            plot_type=plot_type,
            plot_smooth=plot_smooth,
            bounds=bounds,
            derivatives=False,
            intervals=intervals,
            show_all_bounds=show_all_bounds,
            figsize=figsize)
        
    def plot_intervals(self, plot_style: str = 'scatter', 
                       show_data_points: bool = True, 
                       show_grid: bool = True, 
                       figsize: Any = (12, 8)):
        """
        Create specialized visualization focusing exclusively on interval analysis results.
        
        This method generates plots specifically designed to showcase interval detection
        results, providing clear visual separation between tolerance, typical, and atypical
        intervals. It offers multiple visualization styles optimized for interval analysis
        presentation and interpretation.
        
        Parameters
        ----------
        plot_style : str, default='scatter'
            Visualization style for interval display. Options include:
            - 'scatter': Data points with interval region highlighting
            - 'line': Line plot connecting interval boundaries
            - 'smooth': Smoothed curve showing interval regions
        
        show_data_points : bool, default=True
            Whether to overlay original data points on the interval plot.
            When True, shows actual data distribution within detected intervals.
            Set to False for cleaner interval visualization without data overlay.
        
        show_grid : bool, default=True
            Whether to display grid lines for better interval boundary reading.
            Grid lines help in precise identification of interval limits and
            improve overall plot readability for quantitative analysis.
        
        figsize : tuple or Any, default=(12, 8)
            Figure size as (width, height) in inches. Can accept various formats
            depending on the plotting backend. Larger sizes provide more detail
            for complex interval structures.
        
        Returns
        -------
        None
            Displays the interval-focused plot using matplotlib. The visualization
            emphasizes interval boundaries, regions, and relationships between
            different interval types.
        
        Raises
        ------
        RuntimeError
            If the model has not been fitted yet or if interval detection failed.
            Call fit() method successfully before plotting intervals.
        
        ValueError
            If plot_style is not supported or if figsize format is invalid.
        
        Examples
        --------
        >>> import numpy as np
        >>> from machinegnostics.magcal import IntervalAnalysisEGDF
        >>> 
        >>> # Standard interval visualization
        >>> data = np.array([1, 2, 3, 5, 8, 9, 15, 20, 25])
        >>> ia = IntervalAnalysisEGDF(data=data, get_clusters=True)
        >>> ia.fit(plot=False)
        >>> ia.plot_intervals(plot_style='scatter', show_data_points=True)
        >>> 
        >>> # Clean interval boundaries without data overlay
        >>> ia.plot_intervals(plot_style='bar', show_data_points=False, show_grid=True)
        >>> 
        >>> # Area visualization for interval regions
        >>> ia.plot_intervals(plot_style='area', figsize=(15, 10))
        >>> 
        >>> # Detailed analysis with all features
        >>> ia.plot_intervals(
        ...     plot_style='scatter',
        ...     show_data_points=True,
        ...     show_grid=True,
        ...     figsize=(14, 9)
        ... )
        """
        self._plot_egdf_intv(
            plot_style=plot_style,
            show_data_points=show_data_points,
            show_grid=show_grid,
            figsize=figsize)
        