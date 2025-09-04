'''
Marginal Analysis ELDF

Author: Nirmal Parmar
Machine Gnostics
'''

import numpy as np
from machinegnostics.magcal.gdf.base_el_ma import BaseMarginalAnalysisELDF

class MarginalAnalysisELDF(BaseMarginalAnalysisELDF):
    """
    Marginal Analysis for Estimating Local Distribution Function (ELDF) with advanced clustering capabilities.
    
    This class performs comprehensive marginal cluster analysis on data samples to identify critical boundaries
    and intervals that characterize the underlying local data distribution. It provides a complete toolkit
    for local density analysis, boundary detection, and cluster identification with intuitive methods
    for fitting and visualization.
    
    ### Key Features:
    
    **Local Boundary Analysis:**
    
    1. **LB and UB (Lower/Upper Bounds)**: Statistical boundaries of the local distribution optimized during fitting.
       These define the effective support where significant local probability density exists.
    
    2. **DLB and DUB (Data Lower/Upper Bounds)**: Actual minimum and maximum values in the data sample.
       These represent the observed range and serve as hard constraints for local analysis.
    
    3. **CLB and CUB (Cluster Lower/Upper Bounds)**: Boundaries of the main local data cluster.
       Critical for ELDF analysis as they capture local concentration patterns and identify
       the primary region of interest in the local distribution.

    4. **Z0 (Mode)**: Point where local PDF reaches its global maximum. This is the most important
       parameter for local distribution analysis, representing the peak of local probability density.
    
    **ELDF-Specific Characteristics:**
    - **Local density focus**: All analysis is based on local probability density rather than cumulative probability
    - **No sample bounds**: Unlike EGDF, ELDF doesn't compute LSB/USB as they're not relevant for local analysis
    - **Enhanced clustering**: CLB/CUB bounds are primary outputs for understanding local data structure
    - **Mode-centric analysis**: Z0 is the central parameter for local distribution characterization
    
    ### Primary Use Cases:
    
    - **Local Quality Control**: Setting control limits based on local process characteristics
    - **Anomaly Detection**: Identifying outliers relative to local data patterns
    - **Peak Analysis**: Understanding modes and local maxima in data distributions
    - **Density-Based Clustering**: Segmenting data based on local concentration patterns
    - **Process Monitoring**: Real-time assessment of local process behavior
    - **Hotspot Analysis**: Detecting regions of high activity or concentration

    ### Parameters:

    data : np.ndarray
        Input data array for local marginal analysis. Must be a 1D numpy array with numerical values.
        Empty arrays or arrays containing only NaN values will raise an error.
        
    early_stopping_steps : int, default=10
        Number of consecutive optimization steps without improvement before stopping.
        Higher values allow more thorough optimization but increase computation time.
        Must be a positive integer.
        
    cluster_threshold : float, default=0.05
        Threshold for cluster detection as fraction of maximum PDF value.
        Lower values (0.01-0.02) detect more subtle clusters; higher values (0.1-0.2) 
        focus on prominent clusters. Critical parameter for CLB/CUB estimation.
        
    get_clusters : bool, default=True
        Whether to perform cluster analysis and compute CLB/CUB bounds.
        Highly recommended for ELDF analysis. Set to False only for basic local fitting.
        
    DLB : float, optional
        Data Lower Bound override. If None, inferred from data minimum.
        Use to set theoretical minimum for the local distribution.
        
    DUB : float, optional
        Data Upper Bound override. If None, inferred from data maximum.
        Use to set theoretical maximum for the local distribution.
        
    LB : float, optional
        Lower Probable Bound override for local distribution support.
        Defines the practical lower limit where significant local density exists.
        
    UB : float, optional
        Upper Probable Bound override for local distribution support.
        Defines the practical upper limit where significant local density exists.
        
    S : float or 'auto', default='auto'
        Scale parameter for local distribution. When 'auto', estimated from data.
        When float, used as fixed scale. Critical for local density estimation quality.
    
    varS : bool, default=False
        Whether to allow variable scale (S) during optimization.
        If True, S is optimized; if False, S remains fixed. Need to keep S == 'auto' if varS is True.
        
    z0_optimize : bool, default=True
        Whether to use advanced optimization for Z0 (mode) estimation.
        Provides sub-point precision for mode location, important for local analysis.
        
    tolerance : float, default=1e-6
        Numerical tolerance for optimization convergence. Smaller values provide
        higher precision but may require more computation time.
        
    data_form : str, default='a'
        Data processing form:
        - 'a': Additive (linear) - standard for most local analysis
        - 'm': Multiplicative (log-transformed) - for multiplicative processes
        
    n_points : int, default=1000
        Number of points for smooth curve generation. Higher values provide
        smoother visualizations but require more computation. Must be positive.
        
    homogeneous : bool, default=True
        Whether to assume data homogeneity. Affects optimization strategy.
        Set to False for data with multiple distinct local patterns.
        
    catch : bool, default=True
        Whether to store detailed results and enable plotting capabilities.
        Must be True for accessing .params, .plot(), and detailed analysis results.
        
    weights : np.ndarray, optional
        Sample weights for weighted local analysis. Must match data length.
        Use to emphasize specific regions in local density estimation.
        
    wedf : bool, default=True
        Whether to compute Weighted Empirical Distribution Function alongside ELDF.
        Enhances local analysis when weights are provided.
        
    opt_method : str, default='L-BFGS-B'
        Optimization algorithm for parameter estimation. Default works well for
        most local optimization problems. Other options: 'TNC', 'Powell', 'SLSQP'.
        
    verbose : bool, default=False
        Whether to print detailed progress information during fitting.
        Useful for diagnostics and understanding optimization behavior.
        
    max_data_size : int, default=1000
        Safety limit for data size to prevent memory issues during processing.
        
    flush : bool, default=True
        Whether to flush output streams for real-time progress display.

    ### Attributes (Available After fit()):

    CLB : float
        Cluster Lower Bound - lower boundary of the main local cluster
        
    CUB : float
        Cluster Upper Bound - upper boundary of the main local cluster
        
    z0 : float
        Mode of the local distribution (point of maximum PDF)
        
    main_cluster : np.ndarray
        Data points in the main cluster (between CLB and CUB)
        
    lower_cluster : np.ndarray
        Data points below the main cluster (< CLB)
        
    upper_cluster : np.ndarray
        Data points above the main cluster (> CUB)
        
    is_homogeneous : bool
        Whether the data was determined to be homogeneous
        
    params : dict
        Complete dictionary of all computed parameters and results (when catch=True)
        
    init_eldf : ELDF
        The underlying fitted ELDF object with detailed local distribution information

    ### Examples:

    **Basic Local Analysis:**
    
    >>> import numpy as np
    >>> from machinegnostics.magcal import MarginalAnalysisELDF
    >>> 
    >>> # Data with local concentration patterns
    >>> data = np.array([-10,-9,-8,-0.2,-0.1,0,0.1,0.2,8,9,10])
    >>> 
    >>> # Perform marginal analysis
    >>> ma = MarginalAnalysisELDF(data=data, verbose=True)
    >>> ma.fit()
    >>> 
    >>> # Access key results
    >>> print(f"Local mode (Z0): {ma.z0:.3f}")
    >>> print(f"Main cluster: [{ma.CLB:.3f}, {ma.CUB:.3f}]")
    >>> print(f"Cluster size: {len(ma.main_cluster)} points")
    >>> print(f"Data homogeneous: {ma.is_homogeneous}")

    **Advanced Clustering Analysis:**
    
    >>> # Sensitive cluster detection with visualization
    >>> ma = MarginalAnalysisELDF(
    ...     data=data,
    ...     cluster_threshold=0.02,    # Detect subtle clusters
    ...     z0_optimize=True,          # Precise mode location
    ...     n_points=2000,            # Smooth curves
    ...     verbose=True
    ... )
    >>> 
    >>> # Fit and visualize
    >>> ma.fit(plot=True)
    >>> 
    >>> # Detailed cluster analysis
    >>> print(f"Lower cluster: {len(ma.lower_cluster)} points")
    >>> print(f"Main cluster: {len(ma.main_cluster)} points") 
    >>> print(f"Upper cluster: {len(ma.upper_cluster)} points")
    >>> 
    >>> # Additional visualizations
    >>> ma.plot(plot_type='both', bounds=True)           # ELDF + PDF
    >>> ma.plot(derivatives=True)                        # Derivative analysis

    **Quality Control Application:**
    
    >>> # Process monitoring with specification limits
    >>> ma = MarginalAnalysisELDF(
    ...     data=process_measurements,
    ...     LB=lower_spec_limit,
    ...     UB=upper_spec_limit,
    ...     tolerance=1e-8,           # High precision
    ...     verbose=True
    ... )
    >>> ma.fit()
    >>> 
    >>> # Check process capability
    >>> process_capable = (ma.CLB >= ma.init_eldf.LB and 
    ...                    ma.CUB <= ma.init_eldf.UB)
    >>> print(f"Process within specs: {process_capable}")
    >>> 
    >>> # Monitor process centering
    >>> target_center = (ma.init_eldf.LB + ma.init_eldf.UB) / 2
    >>> centering_error = abs(ma.z0 - target_center)
    >>> print(f"Process centering error: {centering_error:.4f}")

    ### Methods:

    fit(plot=False)
        Fit the ELDF marginal analysis model to the data.
        
        Parameters:
        - plot (bool): Whether to display visualization after fitting
        
        Returns:
        - None (sets all analysis attributes)
        
    plot(plot_type='marginal', plot_smooth=True, bounds=True, derivatives=False, figsize=(12, 8))
        Generate comprehensive visualizations of the local analysis results.
        
        Parameters:
        - plot_type (str): 'marginal', 'eldf', 'pdf', or 'both'
        - plot_smooth (bool): Whether to show smooth interpolated curves
        - bounds (bool): Whether to display boundary lines (LB, UB, CLB, CUB, etc.)
        - derivatives (bool): Whether to show derivative analysis plots
        - figsize (tuple): Figure dimensions (width, height)

    ### Notes:

    - **Always call fit() before accessing analysis results or plotting**
    - **Set catch=True (default) to enable result storage and plotting**
    - **CLB/CUB bounds are the primary outputs for local cluster analysis**
    - **Z0 represents the local mode and is the most critical single parameter**
    - **cluster_threshold significantly affects clustering sensitivity**
    - **For heterogeneous data, set homogeneous=False**
    - **Large n_points values provide smoother visualizations but use more memory**
    - **Weighted analysis (wedf=True) enhances results when weights are meaningful**
    
    ### Raises:

    ValueError
        - Empty or invalid data array
        - Invalid parameter values (negative tolerances, invalid bounds, etc.)
        - Mismatched weights array length
        - Invalid plot_type or other method parameters
        
    RuntimeError
        - ELDF fitting fails to converge
        - Cluster boundary estimation fails
        - Plotting attempted before fitting or with catch=False
        
    OptimizationError
        - Underlying optimization algorithm fails
        - Numerical issues in local density estimation

    ### See Also:

    ELDF : Core Estimating Local Distribution Function class
    DataHomogeneity : Homogeneity testing and cluster boundary estimation
    MarginalAnalysisEGDF : Equivalent analysis for global cumulative distributions
    """
    def __init__(self,
                data: np.ndarray,
                early_stopping_steps: int = 10,
                cluster_threshold: float = 0.05,
                get_clusters: bool = True,
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
                flush: bool = True):
        super().__init__(data=data,
                         early_stopping_steps=early_stopping_steps,
                         cluster_threshold=cluster_threshold,
                         get_clusters=get_clusters,
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
        Fit the ELDF marginal analysis model to the data and perform comprehensive cluster analysis.
        
        This method performs the complete local marginal analysis workflow including:
        1. ELDF model fitting with parameter optimization
        2. Local boundary estimation (LB, UB, DLB, DUB)
        3. Homogeneity testing and cluster detection
        4. Cluster boundary estimation (CLB, CUB) when data is heterogeneous
        5. Z0 (mode) identification with optional sub-point precision
        6. Main cluster extraction and outlier identification
        
        After successful fitting, all analysis results become available through instance attributes
        (CLB, CUB, z0, main_cluster, etc.) and the params dictionary when catch=True.
        
        Parameters:
        -----------
        plot : bool, default=False
            Whether to automatically display the marginal analysis plot after fitting.
            When True, shows ELDF and PDF curves with all detected boundaries and clusters.
            Equivalent to calling fit() followed by plot().
            
            Note: Plotting requires catch=True (default) during initialization.
        
        Returns:
        --------
        None
            This method modifies the instance in-place, setting all analysis attributes.
            Use the instance attributes or params dictionary to access results.
        
        Raises:
        -------
        RuntimeError
            If ELDF optimization fails to converge within the specified tolerance and iterations.
            If cluster boundary estimation fails (when get_clusters=True).
            If Z0 optimization fails to locate the local mode.
            If plotting is requested but catch=False was set during initialization.
            
        ValueError
            If the input data is invalid (empty, all NaN, wrong dimensions).
            If any parameter bounds are inconsistent (e.g., DLB > DUB, LB > UB).
            If weights array doesn't match data length.
            
        OptimizationError
            If the underlying optimization algorithm encounters numerical issues.
            If local density estimation fails due to poor data conditioning.
        
        Examples:
        ---------
        **Basic Fitting:**
        
        >>> import numpy as np
        >>> from machinegnostics.magcal import MarginalAnalysisELDF
        >>> 
        >>> data = np.array([-10,-9,-8,-0.2,-0.1,0,0.1,0.2,8,9,10])
        >>> ma = MarginalAnalysisELDF(data=data, verbose=True)
        >>> 
        >>> # Fit without plotting
        >>> ma.fit()
        >>> 
        >>> # Access results
        >>> print(f"Mode (Z0): {ma.z0:.3f}")
        >>> print(f"Cluster bounds: [{ma.CLB:.3f}, {ma.CUB:.3f}]")
        >>> print(f"Main cluster size: {len(ma.main_cluster)}")
        >>> print(f"Homogeneous: {ma.is_homogeneous}")
        
        **Fitting with Automatic Visualization:**
        
        >>> # Fit and immediately see results
        >>> ma = MarginalAnalysisELDF(data=data, cluster_threshold=0.02, verbose=True)
        >>> ma.fit(plot=True)  # Shows plot automatically
        >>> 
        >>> # Results are immediately available
        >>> print(f"Analysis complete. Found {len(ma.main_cluster)} points in main cluster.")
        
        **Error Handling:**
        
        >>> try:
        ...     # Attempt fitting with potentially problematic data
        ...     problematic_data = np.array([np.nan, np.nan, 1, 2])
        ...     ma = MarginalAnalysisELDF(data=problematic_data)
        ...     ma.fit()
        ... except ValueError as e:
        ...     print(f"Data validation error: {e}")
        ... except RuntimeError as e:
        ...     print(f"Fitting failed: {e}")
        
        **Batch Processing Workflow:**
        
        >>> datasets = [data1, data2, data3]
        >>> results = []
        >>> 
        >>> for i, dataset in enumerate(datasets):
        ...     ma = MarginalAnalysisELDF(data=dataset, verbose=False)
        ...     
        ...     try:
        ...         ma.fit()
        ...         results.append({
        ...             'dataset_id': i,
        ...             'z0': ma.z0,
        ...             'cluster_bounds': (ma.CLB, ma.CUB),
        ...             'main_cluster_size': len(ma.main_cluster),
        ...             'homogeneous': ma.is_homogeneous
        ...         })
        ...     except Exception as e:
        ...         print(f"Dataset {i} failed: {e}")
        
        Notes:
        ------
        - **Must be called before accessing any analysis results**
        - **Sets _fitted flag to True upon successful completion**
        - **All instance attributes (CLB, CUB, z0, etc.) are populated during fitting**
        - **params dictionary is updated with complete results when catch=True**
        - **Subsequent calls to fit() will re-run the entire analysis**
        - **For large datasets, consider setting verbose=True to monitor progress**
        - **cluster_threshold parameter significantly affects CLB/CUB detection sensitivity**
        
        See Also:
        ---------
        plot() : Generate detailed visualizations of fitting results
        """
        # Fit the model to the data
        self._fit_eldf(plot=plot)
    
    def plot(self, 
             plot_type: str = 'marginal', 
             plot_smooth: bool = True, 
             bounds: bool = True, 
             figsize: tuple = (12, 8)):
        """
        Generate comprehensive visualizations of the ELDF marginal analysis results.
        
        This method creates professional-quality plots showing the local distribution analysis
        results including ELDF curves, PDF curves, all detected boundaries, cluster bounds,
        and marginal points. Multiple plot types are available for different analysis needs.
        
        The visualization uses a dual y-axis approach with ELDF on the primary axis (blue)
        and PDF on the secondary axis (red), with all boundaries and marginal points clearly
        labeled and color-coded for easy interpretation.
        
        Parameters:
        -----------
        plot_type : str, default='marginal'
            Type of visualization to generate:
            
            - 'marginal': Complete marginal analysis view showing both ELDF and PDF with 
              all boundaries, cluster bounds, and marginal points. Recommended for most users.
            - 'eldf': Focus on ELDF curve only with boundaries and marginal points.
              Useful for understanding local distribution characteristics.
            - 'pdf': Focus on PDF curve only with boundaries and marginal points.
              Useful for density analysis and peak detection.
            - 'both': Equivalent to 'marginal' - shows both ELDF and PDF curves.
              
        plot_smooth : bool, default=True
            Whether to display smooth interpolated curves alongside discrete points.
            
            - True: Shows both discrete points (circles/squares) and smooth curves
              for professional visualization. Uses n_points resolution for smoothness.
            - False: Shows only discrete points connected by lines. Faster rendering
              but less smooth appearance.
              
        bounds : bool, default=True
            Whether to display boundary lines and shaded regions.
            
            - True: Shows all detected boundaries (DLB, DUB, LB, UB, CLB, CUB) as
              vertical lines with labels, plus shaded regions for outlier areas.
            - False: Shows only the ELDF/PDF curves and Z0 marginal point.
              Clean view focusing on the distribution shape.
            
        figsize : tuple, default=(12, 8)
            Figure dimensions as (width, height) in inches.
            Larger sizes provide more detail but use more screen space.
            Recommended range: (8, 6) to (16, 12).
        
        Returns:
        --------
        None
            Displays the plot using matplotlib. No return value.
        
        Raises:
        -------
        RuntimeError
            If fit() has not been called yet (no analysis results available).
            If catch=False was set during initialization (plotting disabled).
            If the underlying ELDF object is not properly fitted.
            
        ValueError
            If plot_type is not one of the valid options.
            If figsize is not a tuple of two positive numbers.
            If derivatives=True but derivative calculation fails.
            
        AttributeError
            If required plotting data is missing from the fitted ELDF object.
        
        Examples:
        ---------
        **Basic Marginal Analysis Plot:**
        
        >>> import numpy as np
        >>> from machinegnostics.magcal import MarginalAnalysisELDF
        >>> 
        >>> data = np.array([-10,-9,-8,-0.2,-0.1,0,0.1,0.2,8,9,10])
        >>> ma = MarginalAnalysisELDF(data=data, verbose=True)
        >>> ma.fit()
        >>> 
        >>> # Standard marginal analysis visualization
        >>> ma.plot()  # Shows ELDF + PDF with all boundaries
        
        **Customized Visualizations:**
        
        >>> # Focus on PDF for density analysis
        >>> ma.plot(plot_type='pdf', bounds=True, figsize=(10, 6))
        >>> 
        >>> # Clean ELDF view without boundaries
        >>> ma.plot(plot_type='eldf', bounds=False, plot_smooth=True)
        >>> 
        >>> # High-resolution smooth curves
        >>> ma_hires = MarginalAnalysisELDF(data=data, n_points=2000)
        >>> ma_hires.fit()
        >>> ma_hires.plot(plot_smooth=True, figsize=(14, 8))
        
        **Presentation-Ready Plots:**
        
        >>> # Large, high-quality plot for presentations
        >>> ma = MarginalAnalysisELDF(
        ...     data=data, 
        ...     n_points=2000,      # High resolution
        ...     cluster_threshold=0.02,  # Sensitive clustering
        ...     verbose=False       # Clean output
        ... )
        >>> ma.fit()
        >>> ma.plot(
        ...     plot_type='marginal',
        ...     plot_smooth=True,
        ...     bounds=True,
        ...     figsize=(16, 10)    # Large size
        ... )
        
        **Batch Visualization:**
        
        >>> # Compare multiple datasets
        >>> datasets = [data1, data2, data3]
        >>> 
        >>> for i, dataset in enumerate(datasets):
        ...     ma = MarginalAnalysisELDF(data=dataset, verbose=False)
        ...     ma.fit()
        ...     
        ...     # Create subplot or separate figures
        ...     ma.plot(plot_type='marginal', bounds=True, 
        ...             figsize=(12, 8))
        ...     plt.title(f'Dataset {i+1} - Local Marginal Analysis')
        ...     plt.show()
        
        Plot Elements:
        --------------
        **Colors and Lines:**
        - Blue: ELDF curve and points
        - Red: PDF curve and points  
        - Light Blue: WEDF points (when available)
        - Green: DLB (Data Lower Bound)
        - Orange: DUB (Data Upper Bound) and CLB/CUB (Cluster bounds)
        - Purple: LB (Lower Bound)
        - Brown: UB (Upper Bound)
        - Magenta: Z0 (Mode) - dash-dot line
        
        **Shaded Regions:**
        - Light purple: Lower outlier region (DLB to LB)
        - Light brown: Upper outlier region (UB to DUB)
        
        **Line Styles:**
        - Solid: Primary boundaries (DLB, DUB)
        - Dashed: Probable boundaries (LB, UB) and cluster bounds (CLB, CUB)
        - Dash-dot: Z0 mode line
        
        Notes:
        ------
        - **Requires successful fit() call before plotting**
        - **catch=True must be set during initialization for plotting to work**
        - **Larger n_points values create smoother curves but use more memory**
        - **All boundary lines include value labels for easy interpretation**
        - **Plot automatically sets appropriate axis limits based on data range**
        - **Grid is enabled for easy value reading**
        - **Legend includes all visible elements with their values**
        
        See Also:
        ---------
        fit() : Perform the marginal analysis before plotting
        """
        # Plot the results
        self._plot_eldf(plot_type=plot_type, plot_smooth=plot_smooth, bounds=bounds, derivatives=False, figsize=figsize)