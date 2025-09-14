"""
EGDF - Estimating Global Distribution Function.

Author: Nirmal Parmar
Machine Gnostics
"""

import numpy as np
from machinegnostics.magcal.gdf.base_egdf import BaseEGDF

class EGDF(BaseEGDF):
    """
    EGDF - Estimating Global Distribution Function.
    
    A comprehensive class for estimating and analyzing global distribution functions for given data.
    This class provides methods to fit distribution functions and visualize results with optional bounds and weighting capabilities.

    The EGDF class supports both additive and multiplicative data forms and can handle bounded and
    unbounded data distributions. It provides automatic parameter estimation and flexible 
    visualization options for distribution analysis.

    The Estimating Global Distribution Function (EGDF) is a gnostic-probabilistic model that estimates the underlying distribution of data points while accounting for various constraints and bounds. 
    It uses gnostic optimization techniques to find the best-fitting parameters and can handle weighted data for improved accuracy in specific applications.

    Key Features:
        - Automatic parameter estimation with customizable bounds
        - Support for weighted data points
        - Multiple data processing forms (additive/multiplicative)
        - Comprehensive visualization capabilities
        - Robust optimization with multiple solver options
        - Memory-efficient processing for large datasets

    Attributes:
        DLB (float): Data Lower Bound - absolute minimum value the data can take.
        DUB (float): Data Upper Bound - absolute maximum value the data can take.
        LB (float): Lower Probable Bound - practical lower limit for the distribution.
        UB (float): Upper Probable Bound - practical upper limit for the distribution.
        S (float or str): Scale parameter for the distribution. Set to 'auto' for automatic estimation.
        z0_optimize (bool): Whether to optimize the location parameter z0 during fitting (default: True).
        data_form (str): Form of the data processing:
            - 'a': Additive form (default) - treats data linearly
            - 'm': Multiplicative form - applies logarithmic transformation
        n_points (int): Number of points to generate in the distribution function (default: 500).
        catch (bool): Whether to store intermediate calculated values (default: True).
        weights (np.ndarray): Prior weights for data points. If None, uniform weights are used.
        wedf (bool): Whether to use Weighted Empirical Distribution Function (default: True) OR KS (Kolmogorov-Smirnov) Points (False).
        opt_method (str): Optimization method for parameter estimation (default: 'L-BFGS-B').
        tolerance (float): Convergence tolerance for optimization (default: 1e-9).
        verbose (bool): Whether to print detailed progress information (default: False).
        params (dict): Dictionary storing fitted parameters and results after fitting.
        homogeneous (bool): To indicate data homogeneity (default: True).
        max_data_size (int): Maximum data size for smooth EGDF generation (default: 1000).
        flush (bool): Whether to flush large arrays (default: True).

    Methods:
        fit(data): Fit the Estimating Global Distribution Function to the data.
        plot(plot_smooth=True, plot='gdf', bounds=False, extra_df=True, figsize=(12,8)): 
            Visualize the fitted distribution.
    
    Examples:
        Basic usage with default parameters:
        >>> import numpy as np
        >>> from machinegnostics.magcal import EGDF
        >>> 
        >>> # Stack Loss example data
        >>> data = np.array([ -13.5, 0, 1. ,   2. ,   3. ,   4. ,   5. ,   6. ,   7. ,   8. , 9. ,  10.,])
        >>> egdf = EGDF()
        >>> egdf.fit(data)
        >>> egdf.plot()
        >>> 
        >>> # Access fitted parameters
        >>> print(f"Fitted parameters: {egdf.params}")
            
    Workflow:
        1. Initialize EGDF with desired parameters (no data required)
        2. Call fit(data) to estimate the distribution parameters
        3. Use plot() to visualize the results
        
        >>> egdf = EGDF(DLB=0, UB=100)  # Step 1: Initialize
        >>> egdf.fit(data)              # Step 2: Fit with data
        >>> egdf.plot(bounds=True)      # Step 3: Visualize
    
    Performance Tips:
        - Use data_form='m' for multiplicative/log-normal data
        - Set appropriate bounds to improve convergence
        - Use catch=False for large datasets to save memory
        - Adjust n_points based on visualization needs vs. performance
        - Use verbose=True to monitor optimization progress
        - For repeated analysis, save fitted parameters and reuse
    
    Common Use Cases:
        - Risk analysis and reliability engineering
        - Quality control and process optimization  
        - Financial modeling and market analysis
        - Environmental data analysis
        - Biostatistics and epidemiological studies
        - Engineering failure analysis
        - Business intelligence and forecasting
    
    Notes:
        - Bounds (DLB, DUB, LB, UB) are optional but can improve estimation accuracy 
          for specific datasets or applications.
        - When S='auto', the scale parameter is automatically estimated from the data
        - The weights array must have the same length as the data array
        - Setting catch=False can save memory for large datasets but prevents access 
          to intermediate results or detailed plots
        - The verbose flag provides detailed output during fitting, useful for debugging 
          or understanding the optimization process
        - The optimization process may take longer for larger datasets or complex distributions
        - Different optimization methods may work better for different types of data
        
    Raises:
        ValueError: If data array is empty or contains invalid values.
        ValueError: If weights array length doesn't match data array length.
        ValueError: If bounds are specified incorrectly (e.g., LB > UB).
        ValueError: If invalid parameters are provided (negative tolerance, invalid data_form, etc.).
        RuntimeError: If the fitting process fails to converge.
        OptimizationError: If the optimization algorithm encounters numerical issues.
        ImportError: If required dependencies (matplotlib for plotting) are not available.
        
    """
    
    def __init__(self,
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
                wedf: bool = True,
                opt_method: str = 'L-BFGS-B',
                verbose: bool = False,
                max_data_size: int = 1000,
                flush: bool = True):
        """
        Initialize the EGDF (Estimating Global Distribution Function) class.

        This constructor sets up all the necessary parameters and configurations for estimating
        a global distribution function from data. It validates input parameters and prepares 
        the instance for subsequent fitting and analysis operations.

        Parameters:
            DLB (float, optional): Data Lower Bound - the absolute minimum value that the data can
                                 theoretically take. If None, will be inferred from data. This is a
                                 hard constraint on the distribution.
            DUB (float, optional): Data Upper Bound - the absolute maximum value that the data can
                                 theoretically take. If None, will be inferred from data. This is a
                                 hard constraint on the distribution.
            LB (float, optional): Lower Probable Bound - the practical lower limit for the distribution.
                                This is typically less restrictive than DLB and represents the expected
                                lower range of the distribution.
            UB (float, optional): Upper Probable Bound - the practical upper limit for the distribution.
                                This is typically less restrictive than DUB and represents the expected
                                upper range of the distribution.
            S (float or str, optional): Scale parameter for the distribution. If 'auto' (default),
                                      the scale will be automatically estimated from the data during
                                      fitting. If a float is provided, it will be used as a fixed
                                      scale parameter.
            tolerance (float, optional): Convergence tolerance for the optimization process.
                                       Smaller values lead to more precise fitting but may require
                                       more iterations. Default is 1e-9.
            data_form (str, optional): Form of data processing. Options are:
                                     - 'a': Additive form (default) - processes data linearly
                                     - 'm': Multiplicative form - applies log transformation for
                                            better handling of multiplicative processes
            n_points (int, optional): Number of points to generate in the final distribution function.
                                    Higher values provide smoother curves but require more computation.
                                    Default is 500. Must be positive integer.
            homogeneous (bool, optional): Whether to assume data homogeneity. Default is True.
                                        Affects internal optimization strategies.
            catch (bool, optional): Whether to store intermediate calculated values during fitting.
                                  Setting to True (default) allows access to detailed results but
                                  uses more memory. Set to False for large datasets to save memory.
            weights (np.ndarray, optional): Prior weights for data points. Must be the same length
                                          as data array when fit() is called. If None, uniform weights 
                                          (all ones) are used. Weights should be positive values.
            wedf (bool, optional): Whether to use Weighted Empirical Distribution Function in
                                 calculations. Default is True. When True, incorporates weights
                                 into the empirical distribution estimation.
            opt_method (str, optional): Optimization method for parameter estimation. Default is
                                      'L-BFGS-B'. Other options include 'SLSQP', 'TNC', etc.
                                      Must be a valid scipy.optimize method name.
            verbose (bool, optional): Whether to print detailed progress information during fitting.
                                    Default is False. When True, provides diagnostic output about
                                    the optimization process.
            max_data_size (int, optional): Maximum size of data for which smooth EGDF generation is allowed.
                                    Maximum data size for processing. Safety limit to prevent excessive memory usage.
            flush (bool, optional): Whether to flush intermediate calculations during processing.
                                  Default is True. May affect memory usage and computation speed.

        Raises:
            ValueError: If n_points is not a positive integer.
            ValueError: If bounds are specified incorrectly (e.g., DLB > DUB or LB > UB).
            ValueError: If data_form is not 'a' or 'm'.
            ValueError: If tolerance is not positive.
            ValueError: If max_data_size is not positive.

        Examples:
            Basic initialization:
            >>> egdf = EGDF()
            
            With custom bounds and weights:
            >>> egdf = EGDF(DLB=0, DUB=5)
            
            Multiplicative form with custom parameters:
            >>> egdf = EGDF(data_form='m')
            
            High precision setup:
            >>> egdf = EGDF(tolerance=1e-12, opt_method='SLSQP', max_data_size=5000)
        
        Notes:
            - The initialization process does not perform any fitting; call fit(data) method afterwards
            - Bounds should be chosen carefully: too restrictive bounds may lead to poor fits
            - For multiplicative data, consider using data_form='m' for better results
            - Large n_points values will slow down plotting but provide smoother visualizations
            - The wedf parameter affects how empirical distributions are calculated
        """
        # parameter
        self.DLB = DLB
        self.DUB = DUB
        self.LB = LB
        self.UB = UB
        self.S = S
        self.z0_optimize = z0_optimize
        self.tolerance = tolerance
        self.data_form = data_form
        self.n_points = n_points
        self.homogeneous = homogeneous
        self.catch = catch
        self.weights = weights
        self.wedf = wedf
        self.opt_method = opt_method
        self.verbose = verbose
        self.max_data_size = max_data_size
        self.flush = flush

    def fit(self, data: np.ndarray, plot: bool = False):
        """
        Fit the Estimating Global Distribution Function to the provided data.

        This method performs the core estimation process for the Estimating Global Distribution Function. 
        The fitting process involves finding the optimal parameters that best describe the underlying 
        distribution of the data while respecting any specified bounds and constraints.

        The EGDF differs from local distribution functions (like ELDF) in that it provides a unique, 
        global representation of the data distribution. Unlike parameterized families of statistical 
        distribution functions, the gnostic distributions have no a priori prescribed form. The scale 
        parameter is automatically optimized to find the best fit, and the EGDF can be used as a model 
        uniquely determined by the data set.

        The method uses numerical optimization techniques to minimize the difference between
        the theoretical distribution and the empirical data. The specific algorithm and
        convergence criteria are controlled by the opt_method and tolerance parameters
        specified during initialization.

        Key Properties of EGDF:
            - Provides unique global distribution for homogeneous data samples
            - Automatically finds optimal scale parameter and bounds
            - Suitable for testing data homogeneity 
            - Robust with respect to outliers
            - Can detect non-homogeneous data through density analysis

        The fitting process:
            1. Validates and preprocesses the data according to the specified data_form
            2. Sets up optimization constraints based on bounds
            3. Transforms data to standard domain for optimization
            4. Runs numerical optimization to find best-fit parameters
            5. Calculates final EGDF and PDF with optimized parameters
            6. Validates and stores the results

        Parameters:
            data (np.ndarray): Input data array for distribution estimation. Must be a 1D numpy array
                             containing numerical values. Empty arrays or arrays with all NaN values
                             will raise an error.
            plot (bool, optional): Whether to automatically plot the fitted distribution after fitting.
                                 Default is False. If True, generates a plot showing the fitted EGDF
                                 and PDF curves along with empirical data comparison.

        Returns:
            None: The method modifies the instance in-place, storing results in self.params
                 and other instance attributes. Access fitted parameters via self.params
                 or use the results() method for comprehensive output.

        Raises:
            RuntimeError: If the optimization process fails to converge within the specified
                         tolerance and maximum iterations.
            ValueError: If the data array is empty, contains only NaN values, or has invalid dimensions.
            ValueError: If weights array is provided but has different length than data array.
            OptimizationError: If the underlying optimization algorithm encounters numerical
                              issues or invalid parameter space.
            ConvergenceError: If the algorithm cannot find a suitable solution.

        Side Effects:
            - Populates self.params with fitted parameters and results
            - Updates internal state variables (_fitted = True)
            - May print progress information if verbose=True
            - Stores intermediate calculations if catch=True
            - Automatically generates plot if plot=True

        Examples:
            Basic fitting:
            >>> egdf = EGDF()
            >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            >>> egdf.fit(data)
            >>> print("Fitting completed")
            >>> print(f"Fitted parameters: {egdf.params}")
            
            Fitting with automatic plotting:
            >>> egdf = EGDF(verbose=True)
            >>> egdf.fit(data, plot=True)  # Will show optimization progress and plot
            
            Accessing results after fitting:
            >>> results = egdf.results()
            >>> print(f"Optimal scale: {results['S_opt']}")
            >>> print(f"Bounds: LB={results['LB']}, UB={results['UB']}")

        Quality Assessment:
            After fitting, check the following for quality assessment:
            - Convergence status in optimization results
            - Parameter values are reasonable for your data
            - No warning messages during verbose output
            - Visual inspection using plot() method
            - Check for homogeneity through density analysis

        Homogeneity Testing:
            The EGDF can be used to test data homogeneity:
            - Homogeneous data: EGDF density has single maximum over infinite domain
            - Non-homogeneous data: Multiple density maxima or negative density values
            - Outliers: Cause local maxima in density near outlier locations
            - Clusters: Appear as separated groups in the distribution

        Notes:
            - This method must be called before using plot() or accessing fitted parameters
            - The fitting process may take several seconds for large datasets
            - If verbose=True was set during initialization, progress information will be printed
            - The quality of the fit depends on the appropriateness of bounds and scale parameters
            - For difficult-to-fit data, consider adjusting tolerance or trying different opt_method
            - Multiple calls to fit() will re-run the optimization process with new data
            - Failed fits may still produce partial results in self.params
            - The EGDF provides a unique representation for each homogeneous data sample

        Performance Tips:
            - Use appropriate bounds to improve convergence speed
            - For large datasets, consider setting catch=False to save memory
            - Increase tolerance slightly for faster fitting of difficult datasets
            - Use verbose=True to monitor optimization progress for debugging

        Troubleshooting:
            If fitting fails:
            - Check data for NaN or infinite values
            - Verify bounds are reasonable (DLB ≤ LB < UB ≤ DUB)
            - Try different optimization methods (SLSQP, TNC)
            - Increase tolerance for difficult datasets
            - Use verbose=True to diagnose optimization issues
            - For non-homogeneous data, consider data preprocessing or clustering

        """
        # Call parent constructor to properly initialize BaseEGDF
        super().__init__(
            data=data,
            DLB=self.DLB,
            DUB=self.DUB,
            LB=self.LB,
            UB=self.UB,
            S=self.S,
            z0_optimize=self.z0_optimize,
            tolerance=self.tolerance,
            data_form=self.data_form,
            n_points=self.n_points,
            catch=self.catch,
            weights=self.weights,
            wedf=self.wedf,
            opt_method=self.opt_method,
            verbose=self.verbose,
            max_data_size=self.max_data_size,
            homogeneous=self.homogeneous,
            flush=self.flush
        )
        self._fit_egdf(plot=plot)

    def plot(self, 
             plot_smooth: bool = True, 
             plot: str = 'both', 
             bounds: bool = False,
             extra_df: bool = True,
             figsize: tuple = (12, 8)):
        """
        Visualize the fitted Estimating Global Distribution Function and related plots.

        This method generates comprehensive visualizations of the fitted distribution function,
        including the main EGDF curve, probability density function (PDF), and optional additional
        distribution functions. The plotting functionality provides insights into the quality
        of the fit and the characteristics of the underlying distribution.

        Multiple plot types and customization options are available to suit different analysis needs.
        The method creates publication-quality plots with proper labels, legends, and formatting.

        For EGDF specifically, the plots help assess:
            - Global distribution characteristics across the entire data range
            - Data homogeneity through density visualization
            - Quality of fit between theoretical and empirical distributions
            - Effect of bounds and constraints on the fitted distribution
            - Identification of potential outliers or clusters

        Parameters:
            plot_smooth (bool, optional): Whether to plot a smooth interpolated curve for the
                                        distribution function. Default is True. When False,
                                        plots discrete points which may be useful for debugging
                                        or analyzing specific data points. Smooth curves provide
                                        better visual appeal but may mask fine details.
            plot (str, optional): Type of plot to generate. Default is 'both'. Options include:
                                - 'gdf': Global Distribution Function (main distribution curve)
                                - 'pdf': Probability Density Function
                                - 'both': Both EGDF and PDF in the same plot
                                Each plot type provides different insights into the distribution.
            bounds (bool, optional): Whether to display bound lines on the plot. Default is False.
                                   When True, shows vertical lines for DLB, DUB, LB, and UB if
                                   they were specified during initialization. Useful for understanding
                                   the constraints applied during fitting and their effect on results.
            extra_df (bool, optional): Whether to include additional distribution functions in
                                     the plot for comparison. Default is True. May include
                                     empirical distribution function (WEDF or KS Points), theoretical 
                                     comparisons, confidence intervals, or goodness-of-fit indicators
                                     depending on the fitting results and available data.
            figsize (tuple, optional): Figure size as (width, height) in inches. Default is (12, 8).
                                     Larger figures provide more detail but use more screen space.
                                     Adjust based on your display requirements and output format.

        Returns:
            None: The method displays the plot(s) using matplotlib. The plot window will appear
                 or the plot will be rendered in the current plotting backend. Plots can be
                 saved using standard matplotlib commands after calling this method.

        Raises:
            RuntimeError: If fit() has not been called before plotting.
            ValueError: If an invalid plot type is specified.
            ImportError: If matplotlib is not available for plotting.
            PlottingError: If there are issues with the plot generation process.
            MemoryError: If plotting large datasets exceeds available memory.

        Side Effects:
            - Creates and displays matplotlib figure(s)
            - May save plots to file if configured in the base class
            - Updates matplotlib's current figure and axes
            - Memory usage increases with plot complexity and data size

        Plot Types Explained:
            - 'gdf': Shows the main fitted EGDF against empirical data, revealing global 
                     distribution characteristics and cumulative probability behavior
            - 'pdf': Displays the probability density function, useful for identifying modes,
                     outliers, and assessing data homogeneity
            - 'both': Combined view showing both EGDF and PDF, providing comprehensive
                      distribution analysis in a single visualization

        Examples:
            Basic plotting after fitting:
            >>> egdf = EGDF()
            >>> egdf.fit(data)
            >>> egdf.plot()
            
            Custom plot with bounds and smooth curve:
            >>> egdf.plot(plot_smooth=True, bounds=True)
            
            Plot probability density function only:
            >>> egdf.plot(plot='pdf', extra_df=False)
            
            Comprehensive visualization with all options:
            >>> egdf.plot(plot='both', bounds=True, extra_df=True, figsize=(16, 10))
            
            Discrete points for large datasets:
            >>> egdf.plot(plot_smooth=False, plot='gdf', figsize=(10, 6))
            
        Customization Tips:
            - Adjust figsize based on your display and output requirements
            - Set plot_smooth=False for very large datasets to improve performance
            - Use bounds=True to visualize the effect of constraints
            - Combine with matplotlib commands for publication-quality figures

        Performance Notes:
            - Smooth plots take longer to generate but look better
            - Large n_points values (set during initialization) slow down plotting
            - For large datasets, consider plot_smooth=False for faster rendering
            - Memory usage scales with data size and plot complexity

        Interpretation Guide:
            EGDF Plots:
            - Good fits show close agreement between fitted and empirical curves
            - Systematic deviations indicate poor model choice or inappropriate bounds
            - Smooth, monotonic curves indicate homogeneous data
            - Steps or plateaus may indicate clustering or non-homogeneity

            PDF Plots:
            - Single peak indicates homogeneous data
            - Multiple peaks suggest clusters or outliers
            - Negative values indicate non-homogeneous data (fitting issues)
            - Sharp spikes near data boundaries may indicate inappropriate bounds

            Bounds Visualization:
            - Bound lines help assess if constraints are too restrictive
            - Data should be well-contained within probable bounds (LB, UB)
            - Wide gaps between data bounds (DLB, DUB) and data may indicate over-constraining

        Notes:
            - The fit() method must be called before plotting
            - Plot appearance can be customized through matplotlib rcParams
            - For large datasets, plotting may take some time
            - The bounds option is only useful if bounds were specified during initialization
            - Different plot types provide different insights into the distribution characteristics
            - Interactive features depend on the matplotlib backend being used
            - Plots remain active until closed or overwritten by new plots
            - Z0 point (if computed) is automatically shown as a vertical line

        Troubleshooting:
            If plotting fails:
            - Ensure fit() was called successfully first
            - Check matplotlib backend is properly configured
            - Verify sufficient memory for large datasets
            - Try simpler plot types if 'both' fails
            - Reduce figsize if display issues occur
            - Check that catch=True was used during initialization for full plotting capability

        """           
        self._plot(plot_smooth=plot_smooth, 
                   plot=plot, 
                   bounds=bounds, 
                   extra_df=extra_df,
                   figsize=figsize)
    
    def results(self) -> dict:
        """
        Retrieve the fitted parameters and comprehensive results from the EGDF fitting process.
    
        This method provides access to all key results obtained after fitting the Estimating Global Distribution Function (EGDF) to the data. 
        It returns a comprehensive dictionary containing fitted parameters, global distribution characteristics, optimization results, 
        and diagnostic information for complete distribution analysis.
    
        The EGDF results focus on global distribution properties, providing a comprehensive view of the entire data distribution
        rather than local characteristics. This makes it particularly valuable for reliability analysis, risk assessment, 
        quality control, and understanding overall distribution behavior across the complete data range.

        Unlike local distribution functions, the EGDF provides a unique representation for each homogeneous data sample,
        with automatically optimized scale parameter and bounds. The results reflect the global nature of the distribution
        and can be used for testing data homogeneity and identifying outliers or clusters.
    
        The results include:
            - Fitted global distribution bounds (DLB, DUB, LB, UB)
            - Optimal scale parameter (S_opt) for global distribution fitting
            - Location parameter (z0) if optimization was enabled
            - EGDF values representing global distribution characteristics
            - PDF values for overall probability density analysis
            - Complete evaluation points covering the full distribution range
            - Weights applied during global distribution fitting
            - Optimization convergence information and performance metrics
            - Error and warning logs from the fitting process
    
        Returns:
            dict: A comprehensive dictionary containing fitted parameters and global distribution results.
                  Primary keys include:
                  
                  Core Global Parameters:
                  - 'DLB': Data Lower Bound used for global distribution fitting
                  - 'DUB': Data Upper Bound used for global distribution fitting
                  - 'LB': Lower Probable Bound for global distribution range
                  - 'UB': Upper Probable Bound for global distribution range
                  - 'S_opt': Optimal scale parameter estimated for global distribution
                  - 'z0': Location parameter (if z0_optimize=True was used)
                  
                  Global Distribution Functions:
                  - 'egdf': EGDF values representing global distribution characteristics
                  - 'pdf': PDF values for comprehensive probability density analysis
                  - 'egdf_points': Points covering full range for smooth EGDF curves
                  - 'pdf_points': Points for complete PDF evaluation across distribution
                  - 'zi': Transformed data points in standardized global domain
                  - 'zi_points': Corresponding evaluation points for global analysis
                  
                  Data and Processing Information:
                  - 'weights': Weights applied to data points (WEDF vs KS approach)
                  - 'wedf': Boolean indicating if Weighted Empirical Distribution Function was used
                  - 'data_form': Data processing form ('a' additive, 'm' multiplicative)
                  - 'n_points': Number of points used for global distribution evaluation
                  - 'homogeneous': Data homogeneity assumption used in fitting
                  
                  Optimization and Quality Metrics:
                  - 'tolerance': Convergence tolerance achieved in optimization
                  - 'opt_method': Optimization method used for parameter estimation
                  - 'max_data_size': Maximum data size limit applied during processing
                  - 'flush': Whether memory flushing was used during computation
                  
                  Diagnostics and Quality Control:
                  - 'errors': List of errors encountered during fitting (if any)
    
        Raises:
            RuntimeError: If fit() has not been called before accessing results.
                         The EGDF model must be successfully fitted before results can be retrieved.
            AttributeError: If internal result structure is missing or corrupted due to fitting failure.
            KeyError: If expected result keys are unavailable, possibly due to incomplete fitting
                     or parameter estimation issues.
            ValueError: If internal state is inconsistent for result retrieval, which may occur
                       if the global distribution fitting process encountered numerical issues.
            MemoryError: If results contain very large arrays that exceed available memory
                        (relevant for large datasets with high n_points values).
    
        Side Effects:
            None. This method provides read-only access to fitting results and does not modify
            the internal state of the EGDF object or trigger any recomputation.
    
        Examples:
            Basic usage after global distribution fitting:
            >>> egdf = EGDF(verbose=True)
            >>> egdf.fit(data)
            >>> results = egdf.results()
            >>> print(f"Global scale parameter: {results['S_opt']:.6f}")
            >>> print(f"Distribution bounds: [{results['LB']:.3f}, {results['UB']:.3f}]")
            
            Accessing global distribution parameters:
            >>> results = egdf.results()
            >>> S_opt = results['S_opt']
            >>> global_bounds = (results['LB'], results['UB'])
            >>> egdf_values = results['egdf']
            >>> pdf_values = results['pdf']
            
            Comprehensive global distribution analysis:
            >>> results = egdf.results()
            >>> print(f"Data bounds: DLB={results['DLB']:.3f}, DUB={results['DUB']:.3f}")
            >>> print(f"Probable bounds: LB={results['LB']:.3f}, UB={results['UB']:.3f}")
            >>> print(f"Global scale: {results['S_opt']:.6f}")
            >>> print(f"Used WEDF: {results.get('wedf', False)}")
            >>> if results.get('errors'):
            ...     print(f"Fitting errors: {len(results['errors'])}")
            
            Homogeneity assessment:
            >>> pdf_values = results['pdf']
            >>> if np.any(pdf_values < 0):
            ...     print("Warning: Negative PDF values detected - data may be non-homogeneous")
            >>> num_modes = len([i for i in range(1, len(pdf_values)-1) 
            ...                  if pdf_values[i] > pdf_values[i-1] and pdf_values[i] > pdf_values[i+1]])
            >>> print(f"Number of density modes: {num_modes}")
    
        Applications:
            EGDF results are particularly valuable for:
            - Reliability engineering and failure analysis across entire operating ranges
            - Risk assessment requiring complete distribution characterization
            - Quality control with global specification limit analysis
            - Financial modeling for portfolio-wide risk assessment
            - Environmental monitoring across complete measurement ranges
            - Process optimization considering full operational envelope
            - Data homogeneity testing and outlier detection
            - Regulatory compliance requiring complete distribution documentation
    
        Interpretation Guide:
            Global Distribution Parameters:
            - 'S_opt': Uniquely determined scale parameter for the data (larger = more spread)
            - 'LB', 'UB': Effective range containing majority of probability mass
            - 'DLB', 'DUB': Hard limits representing absolute data boundaries
            
            Global Distribution Functions:
            - 'egdf': Shows cumulative probability across entire data range
            - 'pdf': Reveals probability density distribution across full spectrum
            - Smooth transitions indicate well-fitted global distribution
            - Sharp discontinuities may indicate fitting issues or data artifacts
            
            Homogeneity Assessment:
            - Single PDF maximum indicates homogeneous data
            - Multiple maxima suggest outliers or clusters
            - Negative PDF values indicate non-homogeneous data
            - Reasonable parameter values suggest appropriate model selection
    
        Performance Considerations:
            - Results retrieval is immediate and cache-optimized
            - Large n_points values increase result array sizes
            - Global distribution evaluation covers broader ranges than local methods
            - Memory usage scales with data size and evaluation point density
    
        Comparison with ELDF:
            EGDF Results Focus:
            - Global distribution characteristics across entire data range
            - Unique, automatically optimized parameters
            - Complete probability mass distribution
            - Comprehensive reliability and risk metrics
            - Data homogeneity testing capabilities
            
            ELDF Results Focus:
            - Local distribution characteristics around critical points
            - Flexible scale parameter for detailed analysis
            - Localized probability concentration analysis
            - Peak detection and modal analysis
    
        Notes:
            - fit() must complete successfully before accessing results
            - Global distribution fitting provides unique parameter values
            - Results structure is consistent regardless of optimization method used
            - WEDF vs KS point selection affects empirical distribution comparison
            - If catch=False was used, some detailed intermediate results may be unavailable
            - All numeric results use appropriate precision for global distribution analysis
            - Results can be serialized for reproducible analysis and reporting
            - Parameter bounds significantly influence global distribution characteristics
    
        Troubleshooting:
            Incomplete Results:
            - Verify fit() completed without exceptions
            - Check 'errors' list for specific fitting issues
            - Ensure adequate tolerance for global optimization convergence
            
            Unexpected Parameter Values:
            - Very large S_opt may indicate scaling issues or inappropriate bounds
            - Check data preprocessing and bound specification
            - Consider different optimization methods for difficult global distributions
            
            Poor Fit Quality:
            - Examine PDF for negative values indicating non-homogeneity
            - Consider adjusting bounds or using different data_form
            - Increase n_points for better global distribution resolution
            - Use verbose=True during fitting to monitor global optimization progress
            
            Memory Issues:
            - Reduce n_points for large datasets
            - Use flush=True for memory-constrained environments
            - Consider catch=False if detailed results are not needed
        """
        if not self._fitted:
            raise RuntimeError("Must fit EGDF before getting results.")
        
        return self._get_results()