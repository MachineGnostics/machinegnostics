'''
ELDF Interval Analysis Module

Estimating Local Marginal Analysis

Author: Nirmal Parmar
Machine Gnostics
'''

import numpy as np
from machinegnostics.magcal.gdf.base_el_intv import BaseIntervalAnalysisELDF

class IntervalAnalysisELDF(BaseIntervalAnalysisELDF):
    """
    Interval Analysis class for Estimating Local Distribution Functions (ELDF) with advanced extrema detection capabilities.
    
    This class performs comprehensive interval analysis on data samples using local distribution function estimation methods
    to identify critical intervals and extrema bounds that characterize the underlying data distribution. It extends the 
    base ELDF functionality with specialized algorithms for extrema detection and robust interval boundary analysis using
    local distribution estimation approaches.
    
    ### Key Features:
    
    **ELDF Interval Analysis - Extrema Detection:**
    
    1. **Tolerance Interval - Location Parameter Extrema (Z0L, Z0U)**: 
       Extrema bounds for the location parameter Z̃0 under local distribution function optimization.
       These bounds represent the minimum and maximum values the location parameter can achieve
       when introducing data points within the specified range. The bounds demonstrate
       the sensitivity and robustness of the local distribution function estimator.
    
    2. **Typical Data Interval - Extrema Search Bounds (ZL, ZU)**:
       The interval bounds where extrema search is performed for new data points. Within this 
       interval, the algorithm searches for data values that produce minimum/maximum location 
       parameter estimates. This interval defines the search space for extrema detection.
    
    3. **Local Distribution Location Parameter (Z0)**:
       The local distribution function estimate of the location parameter from the original data at maximum probability distribution.
       This serves as the baseline for comparison with extrema bounds and represents the
       optimal location estimate under the current data configuration using local distribution methods.

    ### Use Cases:
    
    - **Sensitivity Analysis**: Understanding how new data points affect parameter estimates
    - **Robustness Assessment**: Evaluating stability of local distribution function estimators
    - **Uncertainty Quantification**: Defining parameter uncertainty through extrema bounds
    - **Outlier Impact Analysis**: Measuring potential impact of extreme observations
    - **Local Confidence Intervals**: Distribution-based alternative to bootstrap confidence intervals
    - **Risk Assessment**: Quantifying parameter estimation risk under data variations
    - **Quality Control**: Establishing distribution-based control limits for processes

    ### Attributes:

    data : np.ndarray
        Input data array (1-dimensional) for interval analysis. Must be a 1D numpy array
        containing numerical values. Empty arrays or arrays with all NaN values
        will raise an error.
        
    estimate_cluster_bounds : bool, default=True
        Whether to estimate cluster bounds during interval analysis. When True,
        performs clustering analysis to identify main data groupings and fits
        ELDF specifically to the main cluster for more robust extrema detection.

    get_clusters : bool, default=True
        Whether to perform cluster analysis during interval detection. When True,
        enables cluster-based interval identification and main cluster extraction
        for heterogeneous data handling. Essential for robust analysis of mixed datasets.
        
    n_points_per_direction : int, default=1000
        Number of search points per direction for extrema detection. Higher values
        provide more precise extrema bounds but require more computation. Controls
        the granularity of the search grid used by the interval engine.
        
    dense_zone_fraction : float, default=0.4
        Fraction of search space allocated to dense sampling around critical regions.
        Controls the proportion of search points concentrated in areas of rapid
        distribution function change. Range typically 0.2 to 0.8.
        
    dense_points_fraction : float, default=0.7
        Fraction of total search points allocated to dense zones. Higher values
        provide better resolution in critical regions but may miss global extrema.
        Must be between 0.1 and 0.9.
        
    convergence_window : int, default=15
        Number of consecutive search steps used for convergence assessment.
        Larger windows provide more stable convergence detection but may slow
        down the search process. Must be positive integer.
        
    convergence_threshold : float, default=1e-7
        Threshold for convergence detection in extrema search. Smaller values
        provide more precise extrema bounds but may require more iterations.
        Specific to interval engine optimization processes.
        
    min_search_points : int, default=30
        Minimum number of search points required for reliable extrema detection.
        Safety parameter to ensure adequate sampling density for convergence.
        Must be positive integer, typically 20-100.
        
    boundary_margin_factor : float, default=0.001
        Margin factor for boundary detection to avoid numerical edge effects.
        Controls the buffer zone around detected boundaries. Smaller values
        provide tighter bounds but may be sensitive to numerical precision.
        
    extrema_search_tolerance : float, default=1e-6
        Numerical tolerance for extrema search convergence criteria.
        Smaller values lead to more precise extrema detection but may require
        more iterations. Specific to local distribution optimization convergence.
        
    early_stopping_steps : int, default=10
        Number of consecutive steps without improvement before stopping optimization.
        Prevents infinite loops and improves efficiency during extrema detection.
        
    cluster_threshold : float, default=0.05
        Threshold for PDF-based cluster detection as fraction of maximum PDF value.
        Lower values detect more subtle clusters. Range typically 0.01 to 0.2.
        Used when get_clusters=True for data heterogeneity analysis.
        
    DLB : float, optional
        Data Lower Bound - the absolute minimum value that the data can theoretically take.
        If None, will be inferred from data minimum. Manual override for distribution lower bound.
        
    DUB : float, optional
        Data Upper Bound - the absolute maximum value that the data can theoretically take.
        If None, will be inferred from data maximum. Manual override for distribution upper bound.
        
    LB : float, optional
        Lower Probable Bound - the practical lower limit for interval analysis.
        Manual override for ELDF lower bound used in extrema computations.
        
    UB : float, optional
        Upper Probable Bound - the practical upper limit for interval analysis.
        Manual override for ELDF upper bound used in extrema computations.
        
    S : float or 'auto', default='auto'
        Scale parameter for the local distribution function. If 'auto' (default), 
        the scale will be automatically estimated from the data during fitting. 
        Affects extrema detection sensitivity and distribution optimization.
        
    varS : bool, default=False
        Whether to use variable scale parameter estimation. When True, allows
        the scale parameter to vary during distribution optimization, potentially
        improving fit quality but increasing computational complexity.
        
    z0_optimize : bool, default=True
        Whether to optimize the location parameter during ELDF fitting.
        When True, finds the optimal estimate of the location parameter using
        local distribution methods. Should typically remain True for proper extrema analysis.
        
    tolerance : float, default=1e-6
        Numerical tolerance for convergence criteria in ELDF fitting algorithms.
        Smaller values lead to more precise parameter estimates but may require 
        more iterations. Affects both initial fitting and extrema optimization.
        
    data_form : str, default='a'
        Form of data processing for interval analysis. Options are:
        - 'a': Additive form (default) - processes data linearly
        - 'm': Multiplicative form - applies log transformation for better handling
               of multiplicative processes in distribution estimation
        
    n_points : int, default=1000
        Number of points to generate for ELDF curve evaluation and visualization.
        Higher values provide smoother curves and more precise interval boundaries
        but require more computation. Must be positive integer.
        
    homogeneous : bool, default=True
        Whether to assume data homogeneity during interval analysis. When False,
        enables clustering analysis for heterogeneous data handling. Affects
        extrema detection strategy and cluster-based fitting decisions.
        
    catch : bool, default=True
        Whether to enable error catching and provide detailed interval analysis results.
        Setting to True (default) allows access to detailed results and interval plotting
        but uses more memory. Required for interval plotting and parameter access.
        
    weights : np.ndarray, optional
        Sample weights for weighted local distribution function analysis. Must be the same 
        length as data array. If None, uniform weights are used. Affects distribution
        computation and extrema detection priorities.
        
    wedf : bool, default=True
        Whether to compute Weighted Estimating Distribution Function (WEDF) for interval analysis.
        When True, incorporates weights into distribution computations and extrema detection.
        
    opt_method : str, default='L-BFGS-B'
        Optimization method for distribution optimization and extrema detection.
        Default is 'L-BFGS-B' which handles bounded optimization well.
        Must be a valid scipy.optimize method that supports bounds constraints.
        
    verbose : bool, default=False
        Whether to print detailed progress information during interval analysis.
        When True, provides diagnostic output about optimization progress,
        convergence status, and extrema detection results.
        
    max_data_size : int, default=1000
        Maximum data size for interval processing. Safety limit to prevent excessive 
        memory usage during extrema detection and distribution optimization.
        Large datasets are automatically subsampled.
        
    flush : bool, default=True
        Whether to flush output streams for real-time progress display during 
        extrema detection. May affect memory usage and computation speed during
        intensive distribution optimization processes.
    
    ### Examples

    Basic ELDF interval analysis:
    
    >>> import numpy as np
    >>> from machinegnostics.magcal import IntervalAnalysisELDF
    >>> 
    >>> # Sample data for extrema analysis
    >>> data = np.array([-10,-9,-8,-0.2,-0.1,0,0.1,0.2,8,9,10])
    >>> 
    >>> # Perform ELDF interval analysis
    >>> ia = IntervalAnalysisELDF(
    ...     data=data, 
    ...     n_points_per_direction=2000,
    ...     convergence_threshold=1e-8,
    ...     verbose=True,
    ...     get_clusters=True
    ... )
    >>> ia.fit()
    >>> 
    >>> # Get detected extrema and intervals
    >>> intervals = ia.get_intervals(decimals=4)
    >>> print("Location parameter extrema:", (intervals['Z0L'], intervals['Z0U']))
    >>> print("Search bounds:", (intervals['ZL'], intervals['ZU']))
    >>> print("Baseline location:", intervals['Z0'])
    >>> 
    >>> # Plot extrema analysis results
    >>> ia.plot()
    
    Advanced usage with clustering:
    
    >>> # Data with potential outliers
    >>> mixed_data = np.concatenate([
    ...     np.random.normal(5, 1, 50),    # main cluster
    ...     np.random.normal(15, 0.5, 5)   # outlier cluster
    ... ])
    >>> 
    >>> # Configure for heterogeneous data
    >>> ia_mixed = IntervalAnalysisELDF(
    ...     data=mixed_data,
    ...     homogeneous=False,
    ...     get_clusters=True,
    ...     cluster_threshold=0.03,
    ...     estimate_cluster_bounds=True,
    ...     verbose=True
    ... )
    >>> ia_mixed.fit(plot=True)
    >>> 
    >>> # Compare extrema with and without clustering
    >>> extrema_mixed = ia_mixed.get_intervals()
    >>> sensitivity = extrema_mixed['Z0U'] - extrema_mixed['Z0L']
    >>> print(f"Parameter sensitivity: {sensitivity:.4f}")

    ### Methods

    fit(plot=False)
        Fit the ELDF Interval Analysis model to the data and detect extrema bounds.

    get_intervals(decimals=4)
        Return dictionary containing all detected intervals and extrema bounds
        with specified precision.
        
    plot(figsize=(12, 8))
        Plot the ELDF analysis results with extrema visualization and
        interval engine output.
    
    ### Notes

    - ELDF interval analysis focuses on local distribution function-based extrema detection
    - The algorithm uses iterative search with dense zone sampling for efficiency
    - Extrema bounds quantify parameter uncertainty under data variations
    - Setting get_clusters=True enables robust analysis of heterogeneous data
    - The interval engine performs bidirectional search for comprehensive coverage
    - Convergence parameters significantly affect precision vs. computation tradeoffs
    - For large datasets, consider increasing max_data_size or reducing n_points_per_direction

    ### Raises

    ValueError
        If data array is empty, contains only NaN values, or has invalid dimensions.
        If weights array is provided but has different length than data array.
        If numerical parameters (tolerance, convergence_threshold, etc.) are invalid.
        If n_points_per_direction is too small for reliable extrema detection.
        
    RuntimeError
        If ELDF fitting fails or interval engine fails to converge.
        If extrema detection cannot find valid bounds within the search parameters.
        If clustering analysis fails when get_clusters=True.

    OptimizationError
        If the underlying distribution optimization encounters numerical issues during
        extrema search or parameter estimation.
    
    ConvergenceError
        If the interval engine fails to converge within the specified tolerance
        and maximum iterations. Consider adjusting convergence parameters.
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
        flush: bool = True,
        early_stopping_steps: int = 10,
        cluster_threshold: float = 0.05,
        estimate_cluster_bounds: bool = True,
        get_clusters: bool = True,
        n_points_per_direction: int = 1000, # intv engine specific
        dense_zone_fraction: float = 0.4,
        dense_points_fraction: float = 0.7,
        convergence_window: int = 15,
        convergence_threshold: float = 1e-7,
        min_search_points: int = 30,
        boundary_margin_factor: float = 0.001,
        extrema_search_tolerance: float = 1e-6,):
            
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
            flush=flush,
            early_stopping_steps=early_stopping_steps,
            cluster_threshold=cluster_threshold,
            estimate_cluster_bounds=estimate_cluster_bounds,
            get_clusters=get_clusters,
            n_points_per_direction=n_points_per_direction,
            dense_zone_fraction=dense_zone_fraction,
            dense_points_fraction=dense_points_fraction,
            convergence_window=convergence_window,
            convergence_threshold=convergence_threshold,
            min_search_points=min_search_points,
            boundary_margin_factor=boundary_margin_factor,
            extrema_search_tolerance=extrema_search_tolerance)
        
    def fit(self, plot: bool = False):
        """
        Fit the ELDF Interval Analysis model to the data and detect location parameter extrema.
        
        This method performs the complete ELDF interval analysis workflow including:
        - Fitting the Estimating Local Distribution Function (ELDF) to the data
        - Detecting extrema bounds (Z0L, Z0U) for the location parameter under data variations
        - Identifying search bounds (ZL, ZU) where extrema detection is performed
        - Computing the baseline location parameter (Z0) from local distribution estimation
        - Optionally performing cluster-based analysis if get_clusters=True for heterogeneous data
        - Running the interval engine with bidirectional search for comprehensive extrema detection
        
        The fitting process uses local distribution function optimization with convergence monitoring to ensure
        robust extrema detection while maintaining computational efficiency through dense zone sampling.
        
        Parameters
        ----------
        plot : bool, default=False
            Whether to automatically plot the results after fitting. When True,
            generates a comprehensive visualization showing the ELDF curve, detected
            extrema bounds, search regions, and interval engine results. Set to False 
            for programmatic use without visualization.
        
        Returns
        -------
        None
            This method modifies the object in-place, storing all fitted parameters
            and detected extrema as instance attributes accessible through
            get_intervals() method.
        
        Raises
        ------
        ValueError
            If the data array is empty, contains only NaN values, or has invalid format.
            If any of the fitting parameters (tolerance, n_points_per_direction, etc.) are invalid.
            If clustering parameters are inconsistent when get_clusters=True.
        
        RuntimeError
            If the ELDF fitting process fails to converge within the specified tolerance.
            If extrema detection cannot find valid bounds due to optimization issues.
            If the interval engine fails during bidirectional search.
        
        ConvergenceError
            If the extrema search algorithm fails to converge within convergence_threshold
            and the specified search parameters.
        
        Examples
        --------
        >>> import numpy as np
        >>> from machinegnostics.magcal import IntervalAnalysisELDF
        >>> 
        >>> # Fit with automatic plotting for visualization
        >>> data = np.array([-10,-9,-8,-0.2,-0.1,0,0.1,0.2,8,9,10])
        >>> ia = IntervalAnalysisELDF(data=data, verbose=True)
        >>> ia.fit(plot=True)
        >>> 
        >>> # Fit without plotting for programmatic use
        >>> ia.fit(plot=False)
        >>> extrema = ia.get_intervals()
        >>> print(f"Parameter extrema: [{extrema['Z0L']:.4f}, {extrema['Z0U']:.4f}]")
        >>> 
        >>> # High-precision extrema detection
        >>> ia_precise = IntervalAnalysisELDF(
        ...     data=data,
        ...     n_points_per_direction=5000,
        ...     convergence_threshold=1e-9,
        ...     extrema_search_tolerance=1e-8
        ... )
        >>> ia_precise.fit()
        
        Notes
        -----
        - The extrema bounds represent parameter sensitivity to hypothetical data variations
        - Convergence is monitored using a sliding window approach for stability
        - Dense zone sampling focuses computational resources on critical distribution regions
        - The method must be called before accessing extrema or plotting results
        - For heterogeneous data, clustering analysis improves extrema detection robustness
        - Computational time scales with n_points_per_direction and convergence requirements
        """
        self._fit_eldf_intv(plot=plot)

    def plot(self, figsize=(12, 8)):
        """
        Plot comprehensive ELDF Interval Analysis results with extrema visualization.
        
        This method generates detailed plots showing the fitted ELDF curve, detected extrema bounds,
        search regions, and interval engine optimization results. It provides visualization of both the
        underlying local distribution function and the extrema detection process for thorough analysis
        and presentation of distribution-based interval estimation results.
        
        Parameters
        ----------
        figsize : tuple, default=(12, 8)
            Figure size as (width, height) in inches. Larger sizes provide more detail
            for complex extrema visualization but consume more memory. Adjust based on
            display requirements and available screen space.
        
        Returns
        -------
        None
            Displays the plot using matplotlib. The visualization shows:
            - ELDF curve with local distribution function fitting
            - Extrema bounds (Z0L, Z0U) highlighting parameter sensitivity
            - Search bounds (ZL, ZU) showing the extrema detection region  
            - Baseline location parameter (Z0) as reference point
            - Interval engine convergence and optimization traces
        
        Raises
        ------
        RuntimeError
            If the model has not been fitted yet. Call fit() method first before plotting.
        
        ValueError
            If figsize contains non-positive values or invalid format.
        
        Examples
        --------
        >>> import numpy as np
        >>> from machinegnostics.magcal import IntervalAnalysisELDF
        >>> 
        >>> # Basic extrema visualization
        >>> data = np.array([-10,-9,-8,-0.2,-0.1,0,0.1,0.2,8,9,10])
        >>> ia = IntervalAnalysisELDF(data=data)
        >>> ia.fit(plot=False)
        >>> ia.plot()
        >>> 
        >>> # Detailed visualization for presentation
        >>> ia.plot(figsize=(15, 10))
        >>> 
        >>> # Compact plot for reports
        >>> ia.plot(figsize=(10, 6))
        
        Notes
        -----
        - The plot automatically adjusts scales based on detected extrema range
        - Extrema bounds are highlighted with distinctive colors and markers
        - Search regions show the scope of the interval engine optimization
        - Convergence traces help assess optimization quality and stability
        - The baseline location parameter provides reference for extrema interpretation
        - Large datasets may show subsampled data points for clarity
        """
        self._plot_eldf_intv(figsize=figsize)

    def get_intervals(self, decimals: int = 4):
        """
        Retrieve all detected intervals and extrema bounds from the fitted ELDF Interval Analysis model.
        
        This method returns a comprehensive dictionary containing all extrema bounds and intervals identified
        during the fitting process, including location parameter extrema, search bounds, baseline estimates,
        and optionally cluster-based bounds for heterogeneous data analysis.
        
        Parameters
        ----------
        decimals : int, default=4
            Number of decimal places to round the interval boundaries and extrema bounds.
            Higher values provide more precision but may include numerical noise from
            optimization processes. Must be non-negative. Typical range is 2-8 depending
            on data scale and precision requirements for local distribution-based estimates.
        
        Returns
        -------
        dict
            Dictionary containing detected extrema and intervals with the following keys:
            
            - 'Z0' : float
                Baseline location parameter from local distribution function estimation
            - 'Z0L' : float
                Lower extrema bound for location parameter under data variations
            - 'Z0U' : float
                Upper extrema bound for location parameter under data variations
            - 'ZL' : float
                Lower search bound where extrema detection was performed
            - 'ZU' : float
                Upper search bound where extrema detection was performed
            - 'tolerance_interval' : tuple
                (Z0L, Z0U) - Parameter extrema bounds as interval
            - 'search_bounds' : tuple
                (ZL, ZU) - Bounds where extrema search was conducted
            - 'parameter_sensitivity' : float
                Width of extrema interval (Z0U - Z0L) indicating parameter robustness
            - 'data_bounds' : tuple
                (DLB, DUB) - Theoretical data bounds used in analysis
            - 'probable_bounds' : tuple
                (LB, UB) - Practical probable bounds from ELDF fitting
            - 'main_cluster' : np.ndarray, optional
                Main cluster data points if get_clusters=True was used
        
        Raises
        ------
        RuntimeError
            If the model has not been fitted yet. Call fit() method first.
        
        ValueError
            If decimals parameter is negative or not an integer.
        
        Examples
        --------
        >>> import numpy as np
        >>> from machinegnostics.magcal import IntervalAnalysisELDF
        >>> 
        >>> # Basic extrema retrieval
        >>> data = np.array([-10,-9,-8,-0.2,-0.1,0,0.1,0.2,8,9,10])
        >>> ia = IntervalAnalysisELDF(data=data)
        >>> ia.fit(plot=False)
        >>> intervals = ia.get_intervals(decimals=4)
        >>> 
        >>> # Access core extrema values
        >>> print("Baseline location:", intervals['Z0'])
        >>> print("Extrema bounds:", (intervals['Z0L'], intervals['Z0U']))
        >>> print("Search region:", (intervals['ZL'], intervals['ZU']))
        >>> 
        >>> # Analyze parameter sensitivity
        >>> sensitivity = intervals['Z0U'] - intervals['Z0L']
        >>> print(f"Parameter sensitivity: {sensitivity:.4f}")
        >>> 
        >>> # Compare with baseline uncertainty
        >>> baseline_position = (intervals['Z0'] - intervals['ZL']) / (intervals['ZU'] - intervals['ZL'])
        >>> print(f"Baseline position in search region: {baseline_position:.3f}")
        >>> 
        >>> # High precision for detailed analysis
        >>> precise_intervals = ia.get_intervals(decimals=6)
        >>> extrema_width = precise_intervals['Z0U'] - precise_intervals['Z0L']
        >>> search_width = precise_intervals['ZU'] - precise_intervals['ZL']
        >>> coverage_ratio = extrema_width / search_width
        >>> print(f"Extrema coverage ratio: {coverage_ratio:.4f}")
        >>> 
        >>> # Access interval engine specific results
        >>> if hasattr(ia, 'intv'):
        ...     engine_results = ia.intv.get_intervals(decimals=decimals)
        ...     print("Interval engine results:", engine_results)
        
        Notes
        -----
        - Extrema bounds (Z0L, Z0U) quantify location parameter sensitivity to data changes
        - Search bounds (ZL, ZU) define the region where extrema detection was performed
        - The baseline location parameter (Z0) serves as the local distribution function estimate
        - Parameter sensitivity (Z0U - Z0L) indicates robustness of the location estimate
        - All extrema are computed through local distribution function optimization processes
        - For heterogeneous data, main cluster information may be included in results
        - Precision depends on convergence parameters used during fitting
        - Missing bounds (when estimation fails) return None values
        """
        return self.intv.get_intervals(decimals=decimals)