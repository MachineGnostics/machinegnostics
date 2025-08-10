"""
EGDF - Estimating Global Distribution Function.

Author: Nirmal Parmar
Machine Gnostics
"""

import numpy as np
from machinegnostics.magcal.gdf.base_egdf import BaseEGDF
from machinegnostics.magcal.gdf.bound_estimator import BoundEstimator

class EGDF(BaseEGDF):
    """
    EGDF - Estimating Global Distribution Function.
    
    A comprehensive class for estimating and analyzing global distribution functions (EGDF).
    This class provides methods to fit distribution functions, visualize results, and perform 
    marginal analysis on data with optional bounds and weighting.

    The EGDF class supports both additive and multiplicative data forms and can handle bounded and
    unbounded data distributions. It provides automatic parameter estimation and flexible 
    visualization options for distribution analysis.

    The Estimating Global Distribution Function (EGDF) is a gnostic-probabilistic model that estimates the underlying distribution of data points while accounting for various constraints and bounds. It uses gnostic optimization techniques to find the best-fitting parameters and can handle weighted data for improved accuracy in specific applications.

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
        wedf (bool): Whether to use Weighted Empirical Distribution Function or KS Points (default: False).
        opt_method (str): Optimization method for parameter estimation (default: 'L-BFGS-B').
        tolerance (float): Convergence tolerance for optimization (default: 1e-3).
        verbose (bool): Whether to print detailed progress information (default: False).
        params (dict): Dictionary storing fitted parameters and results after fitting.
        homogeneous (bool): Internal flag indicating data homogeneity (default: True).
    
    Methods:
        fit(): Fit the Estimating global distribution function to the data.
        plot(plot_smooth=True, plot='gdf', bounds=False, extra_df=True): Visualize the fitted distribution.
        marginal_analysis(): Perform marginal analysis on the distribution.
    
    Examples:
        Basic usage with default parameters:
        >>> import numpy as np
        >>> from machinegnostics.magcal import EGDF
        >>> # Stack Loss example data
        >>> data = [7, 8, 8, 8, 9, 11, 12, 13, 14, 14, 15, 15, 15, 18, 18, 19, 20, 28, 37, 37, 42]
        >>> data = np.array(data)
        >>> egdf = EGDF(data)
        >>> egdf.fit()
        >>> egdf.plot()
        
        Usage with custom bounds and weights:
        >>> data = np.random.exponential(2, 500)
        >>> weights = np.random.uniform(0.5, 1.5, 500)
        >>> egdf = EGDF(data, DLB=0, DUB=20, LB=0.1, UB=15, weights=weights)
        >>> egdf.fit()
        >>> egdf.plot(bounds=True)
        
        Multiplicative form with custom scale:
        >>> data = np.random.lognormal(0, 0.5, 800)
        >>> egdf = EGDF(data, data_form='m', S=2.0, n_points=200)
        >>> egdf.fit()
        >>> egdf.marginal_analysis()
    
    Notes:
        - Bounds (DLB, DUB, LB, UB) are optional but can improve estimation accuracy for specific datasets or applications.
        - When S='auto', the scale parameter is automatically estimated from the data
        - The weights array must have the same length as the data array
        - Setting catch=False can save memory for large datasets but prevents access to intermediate results or plots
        - The verbose flag provides detailed output during fitting, useful for debugging or understanding the optimization process
        - The optimization process may take longer for larger datasets or complex distributions
        
    Raises:
        ValueError: If data array is empty or contains invalid values.
        ValueError: If weights array length doesn't match data array length.
        ValueError: If bounds are specified incorrectly (e.g., LB > UB).
        OptimizationError: If the fitting process fails to converge.
    """
    
    def __init__(self,
                 data: np.ndarray,
                 DLB: float = None,
                 DUB: float = None,
                 LB: float = None,
                 UB: float = None,
                 S = 'auto',
                 data_form: str = 'a',
                 n_points: int = 500,
                 catch: bool = True,
                 weights: np.ndarray = None,
                 wedf: bool = False,
                 opt_method: str = 'L-BFGS-B',
                 tolerance: float = 1e-3,
                 verbose: bool = False,
                 max_data_size: int = 100):
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
            data_form (str, optional): Form of data processing. Options are:
                                     - 'a': Additive form (default) - processes data linearly
                                     - 'm': Multiplicative form - applies log transformation for
                                            better handling of multiplicative processes
            n_points (int, optional): Number of points to generate in the final distribution function.
                                    Higher values provide smoother curves but require more computation.
                                    Default is 500. Must be positive integer.
            catch (bool, optional): Whether to store intermediate calculated values during fitting.
                                  Setting to True (default) allows access to detailed results but
                                  uses more memory. Set to False for large datasets to save memory.
            weights (np.ndarray, optional): Prior weights for data points. Must be the same length
                                          as data array. If None, uniform weights (all ones) are used.
                                          Weights should be positive values.
            wedf (bool, optional): Whether to use Weighted Empirical Distribution Function in
                                 calculations. Default is False. When True, incorporates weights
                                 into the empirical distribution estimation.
            opt_method (str, optional): Optimization method for parameter estimation. Default is
                                      'L-BFGS-B'. Other options include 'SLSQP', 'TNC', etc.
                                      Must be a valid scipy.optimize method name.
            tolerance (float, optional): Convergence tolerance for the optimization process.
                                       Smaller values lead to more precise fitting but may require
                                       more iterations. Default is 1e-3.
            verbose (bool, optional): Whether to print detailed progress information during fitting.
                                    Default is False. When True, provides diagnostic output about
                                    the optimization process.
            max_data_size (int, optional): Maximum size of data for which smooth EGDF generation is allowed.
                                    Default is 100. If data exceeds this size, smooth generation
                                    is skipped to avoid excessive memory usage.

        Raises:
            ValueError: If data array is empty, contains only NaN values, or has invalid dimensions.
            ValueError: If weights array is provided but has different length than data array.
            ValueError: If n_points is not a positive integer.
            ValueError: If bounds are specified incorrectly (e.g., DLB > DUB or LB > UB).
            ValueError: If data_form is not 'a' or 'm'.
            ValueError: If tolerance is not positive.

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
            >>> egdf = EGDF(data, data_form='m', S=2.0, n_points=1000, verbose=True)
        
        Notes:
            - The initialization process does not perform any fitting; call fit() method afterwards
            - Bounds should be chosen carefully: too restrictive bounds may lead to poor fits
            - For multiplicative data, consider using data_form='m' for better results
            - Large n_points values will slow down plotting but provide smoother visualizations
        """
        
        self.data = data
        self.DLB = DLB
        self.DUB = DUB
        self.LB = LB
        self.UB = UB
        self.S = S
        self.data_form = data_form
        self.n_points = n_points
        self.homogeneous = True
        self.catch = catch
        self.weights = weights if weights is not None else np.ones_like(data)
        self.wedf = wedf
        self.opt_method = opt_method
        self.tolerance = tolerance
        self.verbose = verbose
        self.max_data_size = max_data_size
        self.params = {}

    def fit(self):
        """
        Fit the Estimating Global Distribution Function to the provided data.

        This method performs the core estimation process for the Estimating Global Distribution Function. The fitting process involves finding the optimal
        parameters that best describe the underlying distribution of the data while respecting
        any specified bounds and constraints.

        The method uses numerical optimization techniques to minimize the difference between
        the theoretical distribution and the empirical data. The specific algorithm and
        convergence criteria are controlled by the opt_method and tolerance parameters
        specified during initialization.

        Returns:
            None: The method modifies the instance in-place, storing results in self.params
                 and other instance attributes.

        Raises:
            RuntimeError: If the optimization process fails to converge within the specified
                         tolerance and maximum iterations.
            ValueError: If the data or parameters are invalid for the fitting process.
            OptimizationError: If the underlying optimization algorithm encounters numerical
                              issues or invalid parameter space.

        Examples:
            Basic fitting:
            >>> egdf = EGDF(data)
            >>> egdf.fit()
            >>> print("Fitting completed")
            >>> print(f"Fitted parameters: {egdf.params}")
            
            Fitting with error handling:
            >>> try:
            ...     egdf.fit()
            ...     print("Fitting successful")
            ... except RuntimeError as e:
            ...     print(f"Fitting failed: {e}")

        Notes:
            - This method must be called before using plot() or marginal_analysis()
            - The fitting process may take several seconds for large datasets
            - If verbose=True was set during initialization, progress information will be printed
            - The quality of the fit depends on the appropriateness of bounds and scale parameters
            - For difficult-to-fit data, consider adjusting tolerance or trying different opt_method

        See Also:
            plot(): For visualizing the fitted distribution
            marginal_analysis(): For analyzing the fitted distribution properties
        """
        self._fit()

    def plot(self, 
             plot_smooth: bool = True, 
             plot: str = 'gdf', 
             bounds: bool = False,
             extra_df: bool = True):
        """
        Visualize the fitted Estimating Global Distribution Function and related plots.

        This method generates comprehensive visualizations of the fitted distribution function,
        including the main EGDF curve, empirical data comparison, and optional additional
        distribution functions. The plotting functionality provides insights into the quality
        of the fit and the characteristics of the underlying distribution.

        Multiple plot types and customization options are available to suit different analysis needs.

        Parameters:
            plot_smooth (bool, optional): Whether to plot a smooth interpolated curve for the
                                        distribution function. Default is True. When False,
                                        plots discrete points which may be useful for debugging
                                        or analyzing specific data points.
            plot (str, optional): Type of plot to generate. Default is 'gdf'. Options include:
                                - 'gdf': Global Distribution Function (main distribution curve)
                                - 'pdf': Probability Density Function
                                - 'cdf': Cumulative Distribution Function
                                - 'both': Both PDF and CDF in subplots
                                - 'all': All available plot types
            bounds (bool, optional): Whether to display bound lines on the plot. Default is False.
                                   When True, shows vertical lines for DLB, DUB, LB, and UB if
                                   they were specified during initialization. Useful for understanding
                                   the constraints applied during fitting.
            extra_df (bool, optional): Whether to include additional distribution functions in
                                     the plot for comparison. Default is True. May include
                                     empirical distribution function (Weighted Empirical Distribution Function or KS Points), theoretical comparisons,
                                     or confidence intervals depending on the fitting results.

        Returns:
            None: The method displays the plot(s) using matplotlib. The plot window will appear
                 or the plot will be rendered in the current plotting backend.

        Raises:
            RuntimeError: If fit() has not been called before plotting.
            ValueError: If an invalid plot type is specified.
            ImportError: If matplotlib is not available for plotting.
            PlottingError: If there are issues with the plot generation process.

        Side Effects:
            - Creates and displays matplotlib figure(s)
            - May save plots to file if configured in the base class
            - Updates matplotlib's current figure and axes

        Examples:
            Basic plotting after fitting:
            >>> egdf = EGDF(data)
            >>> egdf.fit()
            >>> egdf.plot()
            
            Custom plot with bounds and smooth curve:
            >>> egdf.plot(plot_smooth=True, bounds=True)
            
            Plot probability density function:
            >>> egdf.plot(plot='pdf', extra_df=False)
            
            Comprehensive visualization:
            >>> egdf.plot(plot='all', bounds=True, extra_df=True)

        Notes:
            - The fit() method must be called before plotting
            - Plot appearance can be customized through matplotlib rcParams
            - For large datasets, plotting may take some time
            - The bounds option is only useful if bounds were specified during initialization
            - Different plot types provide different insights into the distribution characteristics
            - Interactive features depend on the matplotlib backend being used

        See Also:
            fit(): Must be called before plotting
            marginal_analysis(): For numerical analysis of the distribution
            matplotlib.pyplot: For additional plot customization options
        """
        self._plot(plot_smooth=plot_smooth, 
                   plot=plot, 
                   bounds=bounds, 
                   extra_df=extra_df)

    # def marginal_analysis(self):
    #     """
    #     Perform marginal analysis on the fitted Global Distribution Function.

    #     This method conducts detailed statistical analysis of the fitted distribution,
    #     examining marginal properties, parameter sensitivities, and distribution characteristics.
    #     Marginal analysis helps understand how changes in parameters or data subsets affect
    #     the overall distribution and provides insights into the robustness of the fitted model.

    #     The analysis typically includes examination of parameter confidence intervals,
    #     sensitivity analysis, marginal distributions of multi-dimensional parameters,
    #     and assessment of the distribution's behavior at the boundaries and extremes.

    #     Parameters:
    #         None

    #     Returns:
    #         None: The method may print analysis results to the console or store them in
    #              instance attributes. The specific output depends on the implementation
    #              in derived classes or the base class.

    #     Raises:
    #         RuntimeError: If fit() has not been called before performing marginal analysis.
    #         AnalysisError: If the marginal analysis encounters numerical or computational issues.

    #     Side Effects:
    #         - May print detailed analysis results to the console
    #         - May populate additional attributes with analysis results
    #         - May generate additional plots or visualizations
    #         - Updates internal analysis state

    #     Examples:
    #         Basic marginal analysis:
    #         >>> egdf = EGDF(data)
    #         >>> egdf.fit()
    #         >>> egdf.marginal_analysis()
            
    #         Analysis after complex fitting:
    #         >>> data = np.random.multimodal_distribution(1000)
    #         >>> egdf = EGDF(data, DLB=0, DUB=10)
    #         >>> egdf.fit()
    #         >>> egdf.marginal_analysis()  # Analyze parameter sensitivity

    #     Notes:
    #         - This method provides a default implementation that can be overridden in subclasses
    #         - The current implementation serves as a placeholder for custom analysis logic
    #         - Subclasses may provide specific marginal analysis tailored to particular distribution types
    #         - The analysis quality depends on the quality of the initial fit
    #         - For complex distributions, marginal analysis may reveal parameter correlations
    #         - Results should be interpreted in conjunction with goodness-of-fit metrics

    #     Todo:
    #         - Implement parameter sensitivity analysis
    #         - Add confidence interval calculations
    #         - Include marginal distribution plots
    #         - Provide statistical significance tests
    #         - Add bootstrap-based uncertainty quantification

    #     See Also:
    #         fit(): Must be called before marginal analysis
    #         plot(): For visual inspection of the distribution
    #         scipy.stats: For additional statistical analysis tools
    #     """
    #     lsb, usb = self.estimate_bounds(z_values=self.zi_n)