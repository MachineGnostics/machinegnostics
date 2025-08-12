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
        data (np.ndarray): The input dataset used for distribution estimation.
        DLB (float): Data Lower Bound - absolute minimum value the data can take.
        DUB (float): Data Upper Bound - absolute maximum value the data can take.
        LB (float): Lower Probable Bound - practical lower limit for the distribution.
        UB (float): Upper Probable Bound - practical upper limit for the distribution.
        S (float or str): Scale parameter for the distribution. Set to 'auto' for automatic estimation.
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
        fit(): Fit the Estimating Global Distribution Function to the data.
        plot(plot_smooth=True, plot='gdf', bounds=False, extra_df=True, figsize=(12,8)): 
            Visualize the fitted distribution.
    
    Examples:
        Basic usage with default parameters:
        >>> import numpy as np
        >>> from machinegnostics.magcal.gdf import EGDF
        >>> 
        >>> # Stack Loss example data
        >>> data = [7, 8, 8, 8, 9, 11, 12, 13, 14, 14, 15, 15, 15, 18, 18, 19, 20, 28, 37, 37, 42]
        >>> data = np.array(data)
        >>> egdf = EGDF(data)
        >>> egdf.fit()
        >>> egdf.plot()
        >>> 
        >>> # Access fitted parameters
        >>> print(f"Fitted parameters: {egdf.params}")
        
        Usage with custom bounds and weights:
        >>> 
        >>> # Fit with bounds and weights
        >>> # S is automatically estimated
        >>> egdf = EGDF(data, LB=1, UB=150)
        >>> egdf.fit()
        >>> egdf.plot(bounds=True)
        
        Multiplicative form with custom scale:
        >>> # For log-normal or multiplicative processes
        >>> data = np.random.lognormal(0, 0.5, 8)
        >>> egdf = EGDF(data, data_form='m', S=2.0, n_points=200)
        >>> egdf.fit()
        >>> egdf.plot(plot='pdf')
        
        Memory-efficient processing for large datasets:
        >>> # For very large datasets
        >>> large_data = np.random.normal(0, 1, 50000)
        >>> egdf = EGDF(large_data, 
        ...              catch=False,  # Save memory
        ...              n_points=200,
        ...              max_data_size=10000)
        >>> egdf.fit()
        >>> egdf.plot(plot_smooth=False)  # Discrete points for speed
    
    Workflow:
        1. Initialize EGDF with your data and desired parameters
        2. Call fit() to estimate the distribution parameters
        3. Use plot() to visualize the results
        
        >>> egdf = EGDF(data, DLB=0, UB=100)  # Step 1: Initialize
        >>> egdf.fit()                        # Step 2: Fit
        >>> egdf.plot(bounds=True)           # Step 3: Visualize
    
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
                data: np.ndarray,
                DLB: float = None,
                DUB: float = None,
                LB: float = None,
                UB: float = None,
                S = 'auto',
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
        a global distribution function from the provided data. It validates input parameters
        and prepares the instance for subsequent fitting and analysis operations.

        Parameters:
            data (np.ndarray): Input data array for distribution estimation. Must be a 1D numpy array
                             containing numerical values. Empty arrays or arrays with all NaN values
                             will raise an error.
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
                                          as data array. If None, uniform weights (all ones) are used.
                                          Weights should be positive values.
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
                                         Default is 1000. If data exceeds this size, smooth generation
                                         may be skipped to avoid excessive memory usage.
            flush (bool, optional): Whether to flush intermediate calculations during processing.
                                  Default is True. May affect memory usage and computation speed.

        Raises:
            ValueError: If data array is empty, contains only NaN values, or has invalid dimensions.
            ValueError: If weights array is provided but has different length than data array.
            ValueError: If n_points is not a positive integer.
            ValueError: If bounds are specified incorrectly (e.g., DLB > DUB or LB > UB).
            ValueError: If data_form is not 'a' or 'm'.
            ValueError: If tolerance is not positive.
            ValueError: If max_data_size is not positive.

        Examples:
            Basic initialization:
            >>> data = np.array([1, 2, 3, 4, 5])
            >>> egdf = EGDF(data)
            
            With custom bounds and weights:
            >>> data = np.array([0.5, 1.2, 2.3, 1.8, 3.1])
            >>> weights = np.array([1.0, 0.8, 1.2, 0.9, 1.1])
            >>> egdf = EGDF(data, DLB=0, DUB=5, weights=weights)
            
            Multiplicative form with custom parameters:
            >>> data = np.random.lognormal(0, 1, 100)
            >>> egdf = EGDF(data, data_form='m')
            
            High precision setup:
            >>> egdf = EGDF(data, tolerance=1e-12, opt_method='SLSQP', max_data_size=5000)
        
        Notes:
            - The initialization process does not perform any fitting; call fit() method afterwards
            - Bounds should be chosen carefully: too restrictive bounds may lead to poor fits
            - For multiplicative data, consider using data_form='m' for better results
            - Large n_points values will slow down plotting but provide smoother visualizations
            - The wedf parameter affects how empirical distributions are calculated
        """
        
        # Call parent constructor to properly initialize BaseEGDF
        super().__init__(
            data=data,
            DLB=DLB,
            DUB=DUB,
            LB=LB,
            UB=UB,
            S=S,
            tolerance=tolerance,
            data_form=data_form,
            n_points=n_points,
            catch=catch,
            weights=weights,
            wedf=wedf,
            opt_method=opt_method,
            verbose=verbose,
            max_data_size=max_data_size,
            homogeneous=homogeneous,
            flush=flush
        )

    def fit(self):
        """
        Fit the Estimating Global Distribution Function to the provided data.

        This method performs the core estimation process for the Estimating Global Distribution Function. 
        The fitting process involves finding the optimal parameters that best describe the underlying 
        distribution of the data while respecting any specified bounds and constraints.

        The method uses numerical optimization techniques to minimize the difference between
        the theoretical distribution and the empirical data. The specific algorithm and
        convergence criteria are controlled by the opt_method and tolerance parameters
        specified during initialization.

        The fitting process:
            1. Preprocesses the data according to the specified data_form
            2. Sets up optimization constraints based on bounds
            3. Initializes parameter estimates
            4. Runs numerical optimization to find best-fit parameters
            5. Validates and stores the results

        Returns:
            None: The method modifies the instance in-place, storing results in self.params
                 and other instance attributes. Access fitted parameters via self.params.

        Raises:
            RuntimeError: If the optimization process fails to converge within the specified
                         tolerance and maximum iterations.
            ValueError: If the data or parameters are invalid for the fitting process.
            OptimizationError: If the underlying optimization algorithm encounters numerical
                              issues or invalid parameter space.
            ConvergenceError: If the algorithm cannot find a suitable solution.

        Side Effects:
            - Populates self.params with fitted parameters
            - Updates internal state variables
            - May print progress information if verbose=True
            - Stores intermediate calculations if catch=True

        Examples:
            Basic fitting:
            >>> egdf = EGDF(data)
            >>> egdf.fit()
            >>> print("Fitting completed")
            >>> print(f"Fitted parameters: {egdf.params}")
            
            Monitoring verbose output:
            >>> egdf = EGDF(data, verbose=True, tolerance=1e-6)
            >>> egdf.fit()  # Will print detailed optimization progress
            
            Accessing results after fitting:
            >>> egdf.params  # Contains global distribution parameters

        Quality Assessment:
            After fitting, check the following for quality assessment:
            - Convergence status in optimization results
            - Parameter values are reasonable for your data
            - No warning messages during verbose output
            - Visual inspection using plot() method

        Notes:
            - This method must be called before using plot() or accessing fitted parameters
            - The fitting process may take several seconds for large datasets
            - If verbose=True was set during initialization, progress information will be printed
            - The quality of the fit depends on the appropriateness of bounds and scale parameters
            - For difficult-to-fit data, consider adjusting tolerance or trying different opt_method
            - Multiple calls to fit() will re-run the optimization process
            - Failed fits may still produce partial results in self.params

        Troubleshooting:
            If fitting fails:
            - Check data for NaN or infinite values
            - Verify bounds are reasonable (DLB ≤ LB < UB ≤ DUB)
            - Try different optimization methods
            - Increase tolerance for difficult datasets
            - Use verbose=True to diagnose optimization issues

        """
        self._fit()

    def plot(self, 
             plot_smooth: bool = True, 
             plot: str = 'gdf', 
             bounds: bool = False,
             extra_df: bool = True,
             figsize: tuple = (12, 8)):
        """
        Visualize the fitted Estimating Global Distribution Function and related plots.

        This method generates comprehensive visualizations of the fitted distribution function,
        including the main EGDF curve, data comparison, and optional additional
        distribution functions. The plotting functionality provides insights into the quality
        of the fit and the characteristics of the underlying distribution.

        Multiple plot types and customization options are available to suit different analysis needs.
        The method creates publication-quality plots with proper labels, legends, and formatting.

        Parameters:
            plot_smooth (bool, optional): Whether to plot a smooth interpolated curve for the
                                        distribution function. Default is True. When False,
                                        plots discrete points which may be useful for debugging
                                        or analyzing specific data points. Smooth curves provide
                                        better visual appeal but may mask fine details.
            plot (str, optional): Type of plot to generate. Default is 'gdf'. Options include:
                                - 'gdf': Global Distribution Function (main distribution curve)
                                - 'pdf': Probability Density Function
                                - 'cdf': Cumulative Distribution Function  
                                - 'both': Both PDF and CDF in subplots
                                - 'all': All available plot types in a comprehensive layout
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
            - 'gdf': Shows the main fitted distribution function against empirical data
            - 'pdf': Displays the probability density function (derivative of CDF)
            - 'cdf': Shows the cumulative distribution function (0 to 1 scale)
            - 'both': Side-by-side PDF and CDF for comprehensive view
            - 'all': Multiple panels showing different aspects of the distribution

        Examples:
            Basic plotting after fitting:
            >>> egdf = EGDF(data)
            >>> egdf.fit()
            >>> egdf.plot()
            
            Custom plot with bounds and smooth curve:
            >>> egdf.plot(plot_smooth=True, bounds=True)
            
            Plot probability density function only:
            >>> egdf.plot(plot='pdf', extra_df=False)
            
            Comprehensive visualization with all options:
            >>> egdf.plot(plot='all', bounds=True, extra_df=True, figsize=(16, 10))
            
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
            - Complex plot types ('all') require more computation and memory
            - For large datasets, consider plot_smooth=False for faster rendering

        Interpretation Guide:
            - Good fits show close agreement between fitted and empirical curves
            - Systematic deviations indicate poor model choice or inappropriate bounds
            - Bound lines help assess if constraints are too restrictive
            - Multiple plot types reveal different aspects of distribution behavior

        Notes:
            - The fit() method must be called before plotting
            - Plot appearance can be customized through matplotlib rcParams
            - For large datasets, plotting may take some time
            - The bounds option is only useful if bounds were specified during initialization
            - Different plot types provide different insights into the distribution characteristics
            - Interactive features depend on the matplotlib backend being used
            - Plots remain active until closed or overwritten by new plots

        Troubleshooting:
            If plotting fails:
            - Ensure fit() was called successfully first
            - Check matplotlib backend is properly configured
            - Verify sufficient memory for large datasets
            - Try simpler plot types if 'all' fails
            - Reduce figsize if display issues occur

        """
        self._plot(plot_smooth=plot_smooth, 
                   plot=plot, 
                   bounds=bounds, 
                   extra_df=extra_df,
                   figsize=figsize)