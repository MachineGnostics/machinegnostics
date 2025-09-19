"""
ELDF - Estimating Local Distribution Function.

Author: Nirmal Parmar
Machine Gnostics
"""

import numpy as np
from machinegnostics.magcal.gdf.base_eldf import BaseELDF

class ELDF(BaseELDF):
    """
    ELDF - Estimating Local Distribution Function.

    A comprehensive class for estimating and analyzing local distribution functions for given data.
    This class provides methods to fit local distribution functions and visualize results with optional bounds, weighting capabilities, and advanced Z0 (Gnostic Mean) point estimation.

    The ELDF class supports both additive and multiplicative data forms and can handle bounded and
    unbounded data distributions. It provides automatic parameter estimation and flexible 
    visualization options for local distribution analysis.

    The Estimating Local Distribution Function (ELDF) is a gnostic-probabilistic model that estimates the underlying local distribution of data points while accounting for various constraints and bounds. 
    It uses advanced optimization techniques to find the best-fitting parameters and can handle weighted data for improved accuracy in specific applications. ELDF focuses on local characteristics and provides detailed PDF analysis around critical points, including Z0 estimation.

    Key Features:
        - Automatic parameter estimation with customizable bounds
        - Advanced Z0 point estimation for maximum PDF location
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
        varS (bool): Whether to allow variable scale parameter during optimization (default: False).
        z0_optimize (bool): Whether to optimize the location parameter Z0 during fitting (default: True).
        tolerance (float): Convergence tolerance for optimization (default: 1e-5).
        data_form (str): Form of the data processing:
            - 'a': Additive form (default) - treats data linearly
            - 'm': Multiplicative form - applies logarithmic transformation
        n_points (int): Number of points to generate in the distribution function (default: 1000).
        homogeneous (bool): Whether to assume data homogeneity (default: True).
        catch (bool): Whether to store intermediate calculated values (default: True).
        weights (np.ndarray): Prior weights for data points. If None, uniform weights are used.
        wedf (bool): Whether to use Weighted Empirical Distribution Function (default: True).
        opt_method (str): Optimization method for parameter estimation (default: 'L-BFGS-B').
        verbose (bool): Whether to print detailed progress information (default: False).
        max_data_size (int): Maximum data size for smooth ELDF generation (default: 1000).
        flush (bool): Whether to flush large arrays during processing (default: True).
        params (dict): Dictionary storing fitted parameters and results after fitting.

    Methods:
        fit(data): Fit the Estimating Local Distribution Function to the data.
        plot(plot_smooth=True, plot='eldf', bounds=True, extra_df=True, figsize=(12,8)): 
            Visualize the fitted local distribution.
        results(): Get the fitting results as a dictionary.

    Examples:
        Basic usage with default parameters:
        >>> import numpy as np
        >>> from machinegnostics.magcal import ELDF
        >>> 
        >>> data = np.array([ -13.5, 0, 1. ,   2. ,   3. ,   4. ,   5. ,   6. ,   7. ,   8. , 9. ,  10.,])
        >>> eldf = ELDF()
        >>> eldf.fit(data)
        >>> eldf.plot()
        >>> print(f"Fitted parameters: {eldf.params}")

    Workflow:
        1. Initialize ELDF with desired parameters (no data required)
        2. Call fit(data) to estimate the distribution parameters
        3. Use plot() to visualize the results

        >>> eldf = ELDF(DLB=0, UB=100)  # Step 1: Initialize
        >>> eldf.fit(data)              # Step 2: Fit with data
        >>> eldf.plot(bounds=True)      # Step 3: Visualize

    Performance Tips:
        - Use data_form='m' for multiplicative/log-normal data
        - Set appropriate bounds to improve convergence
        - Use catch=False for large datasets to save memory
        - Adjust n_points based on visualization needs vs. performance
        - Use verbose=True to monitor optimization progress
        - For repeated analysis, save fitted parameters and reuse

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
        """
        Initialize the ELDF (Estimating Local Distribution Function) class.

        This constructor sets up all the necessary parameters and configurations for estimating
        a local distribution function from data. It validates input parameters and prepares 
        the instance for subsequent fitting and analysis operations.

        Parameters:
            DLB (float, optional): Data Lower Bound - the absolute minimum value that the data can
                                 theoretically take. If None, will be inferred from data.
            DUB (float, optional): Data Upper Bound - the absolute maximum value that the data can
                                 theoretically take. If None, will be inferred from data.
            LB (float, optional): Lower Probable Bound - the practical lower limit for the distribution.
            UB (float, optional): Upper Probable Bound - the practical upper limit for the distribution.
            S (float or str, optional): Scale parameter for the distribution. If 'auto' (default),
                                      the scale will be automatically estimated from the data during
                                      fitting. If a float is provided, it will be used as a fixed
                                      scale parameter.
            varS (bool, optional): Whether to allow variable scale parameter during optimization.
            z0_optimize (bool, optional): Whether to optimize the location parameter Z0 during fitting.
            tolerance (float, optional): Convergence tolerance for the optimization process.
            data_form (str, optional): Form of data processing. Options are:
                                     - 'a': Additive form (default)
                                     - 'm': Multiplicative form
            n_points (int, optional): Number of points to generate in the final distribution function.
            homogeneous (bool, optional): Whether to assume data homogeneity.
            catch (bool, optional): Whether to store intermediate calculated values during fitting.
            weights (np.ndarray, optional): Prior weights for data points.
            wedf (bool, optional): Whether to use Weighted Empirical Distribution Function.
            opt_method (str, optional): Optimization method for parameter estimation.
            verbose (bool, optional): Whether to print detailed progress information during fitting.
            max_data_size (int, optional): Maximum size of data for which smooth ELDF generation is allowed.
            flush (bool, optional): Whether to flush intermediate calculations during processing.

        Raises:
            ValueError: If n_points is not a positive integer.
            ValueError: If bounds are specified incorrectly.
            ValueError: If data_form is not 'a' or 'm'.
            ValueError: If tolerance is not positive.
            ValueError: If max_data_size is not positive.

        Examples:
            >>> eldf = ELDF()
            >>> eldf = ELDF(DLB=0, DUB=5)
            >>> eldf = ELDF(data_form='m')
            >>> eldf = ELDF(tolerance=1e-6, opt_method='SLSQP', max_data_size=5000)
        """
        self.DLB = DLB
        self.DUB = DUB
        self.LB = LB
        self.UB = UB
        self.S = S
        self.varS = varS
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
        Fit the Estimating Local Distribution Function to the provided data.

        This method performs the core estimation process for the Estimating Local Distribution Function. 
        The fitting process involves finding the optimal parameters that best describe the underlying 
        local distribution of the data while respecting any specified bounds and constraints.

        The ELDF provides detailed local analysis, including Z0 estimation (location of maximum PDF).
        The method uses numerical optimization techniques to minimize the difference between
        the theoretical local distribution and the empirical data. The specific algorithm and
        convergence criteria are controlled by the opt_method and tolerance parameters
        specified during initialization.

        Key Properties of ELDF:
            - Provides detailed local distribution for data samples
            - Automatically finds optimal scale parameter and bounds
            - Z0 estimation for maximum PDF location
            - Robust with respect to outliers and local density variations

        The fitting process:
            1. Validates and preprocesses the data according to the specified data_form
            2. Sets up optimization constraints based on bounds
            3. Transforms data to standard domain for optimization
            4. Runs numerical optimization to find best-fit parameters
            5. Calculates final ELDF and PDF with optimized parameters
            6. Estimates Z0 point if enabled
            7. Validates and stores the results

        Parameters:
            data (np.ndarray): Input data array for distribution estimation. Must be a 1D numpy array
                             containing numerical values. Empty arrays or arrays with all NaN values
                             will raise an error.
            plot (bool, optional): Whether to automatically plot the fitted distribution after fitting.
                                 Default is False. If True, generates a plot showing the fitted ELDF
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
            >>> eldf = ELDF()
            >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            >>> eldf.fit(data)
            >>> print("Fitting completed")
            >>> print(f"Fitted parameters: {eldf.params}")
            
            Fitting with automatic plotting:
            >>> eldf = ELDF(verbose=True)
            >>> eldf.fit(data, plot=True)  # Will show optimization progress and plot
            
            Accessing results after fitting:
            >>> results = eldf.results()
            >>> print(f"Optimal scale: {results['S_opt']}")
            >>> print(f"Bounds: LB={results['LB']}, UB={results['UB']}")
            >>> print(f"Z0 point: {results['z0']}")
        """
        super().__init__(
            data=data,
            DLB=self.DLB,
            DUB=self.DUB,
            LB=self.LB,
            UB=self.UB,
            S=self.S,
            varS=self.varS,
            z0_optimize=self.z0_optimize,
            tolerance=self.tolerance,
            data_form=self.data_form,
            n_points=self.n_points,
            homogeneous=self.homogeneous,
            catch=self.catch,
            weights=self.weights,
            wedf=self.wedf,
            opt_method=self.opt_method,
            verbose=self.verbose,
            max_data_size=self.max_data_size,
            flush=self.flush
        )
        self._fit_eldf(plot=plot)

    def plot(self, 
             plot_smooth: bool = True, 
             plot: str = 'both', 
             bounds: bool = True,
             extra_df: bool = True,
             figsize: tuple = (12, 8)):
        """
        Visualize the fitted Estimating Local Distribution Function and related plots.

        This method generates comprehensive visualizations of the fitted local distribution function,
        including the main ELDF curve, probability density function (PDF), and optional additional
        distribution functions. The plotting functionality provides insights into the quality
        of the fit and the characteristics of the underlying local distribution.

        Multiple plot types and customization options are available to suit different analysis needs.
        The method creates publication-quality plots with proper labels, legends, and formatting.

        For ELDF specifically, the plots help assess:
            - Local distribution characteristics around critical points
            - Z0 location (maximum PDF)
            - Quality of fit between theoretical and empirical distributions
            - Effect of bounds and constraints on the fitted distribution
            - Identification of potential outliers or clusters

        Parameters:
            plot_smooth (bool, optional): Whether to plot a smooth interpolated curve for the
                                        distribution function. Default is True. When False,
                                        plots discrete points which may be useful for debugging
                                        or analyzing specific data points. Smooth curves provide
                                        better visual appeal but may mask fine details.
            plot (str, optional): Type of plot to generate. Default is 'eldf'. Options include:
                                - 'eldf': Local Distribution Function (main distribution curve)
                                - 'pdf': Probability Density Function
                                - 'both': Both ELDF and PDF in the same plot
                                Each plot type provides different insights into the distribution.
            bounds (bool, optional): Whether to display bound lines on the plot. Default is True.
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
            - 'eldf': Shows the main fitted ELDF against empirical data, revealing local 
                     distribution characteristics and cumulative probability behavior
            - 'pdf': Displays the probability density function, useful for identifying modes,
                     outliers, and assessing local density
            - 'both': Combined view showing both ELDF and PDF, providing comprehensive
                      distribution analysis in a single visualization

        Examples:
            Basic plotting after fitting:
            >>> eldf = ELDF()
            >>> eldf.fit(data)
            >>> eldf.plot()
            
            Custom plot with bounds and smooth curve:
            >>> eldf.plot(plot_smooth=True, bounds=True)
            
            Plot probability density function only:
            >>> eldf.plot(plot='pdf', extra_df=False)
            
            Comprehensive visualization with all options:
            >>> eldf.plot(plot='both', bounds=True, extra_df=True, figsize=(16, 10))
            
            Discrete points for large datasets:
            >>> eldf.plot(plot_smooth=False, plot='eldf', figsize=(10, 6))
        """           
        self._plot(plot_smooth=plot_smooth, 
                   plot=plot, 
                   bounds=bounds, 
                   extra_df=extra_df,
                   figsize=figsize)
    
    def results(self) -> dict:
        """
        Retrieve the fitted parameters and comprehensive results from the ELDF fitting process.

        This method provides access to all key results obtained after fitting the Estimating Local Distribution Function (ELDF) to the data. 
        It returns a comprehensive dictionary containing fitted parameters, local distribution characteristics, optimization results, 
        and diagnostic information for complete distribution analysis.

        The ELDF results focus on local distribution properties, providing a detailed view of the distribution around critical points,
        including Z0 estimation. This makes it particularly valuable for peak detection, modal analysis, and local density estimation.

        The results include:
            - Fitted local distribution bounds (DLB, DUB, LB, UB)
            - Optimal scale parameter (S_opt) for local distribution fitting
            - Location parameter (z0) if optimization was enabled
            - ELDF values representing local distribution characteristics
            - PDF values for local probability density analysis
            - Complete evaluation points covering the local distribution range
            - Weights applied during local distribution fitting
            - Optimization convergence information and performance metrics
            - Error and warning logs from the fitting process

        Returns:
            dict: A comprehensive dictionary containing fitted parameters and local distribution results.

        Raises:
            RuntimeError: If fit() has not been called before accessing results.
                         The ELDF model must be successfully fitted before results can be retrieved.
            AttributeError: If internal result structure is missing or corrupted due to fitting failure.
            KeyError: If expected result keys are unavailable, possibly due to incomplete fitting
                     or parameter estimation issues.
            ValueError: If internal state is inconsistent for result retrieval, which may occur
                       if the local distribution fitting process encountered numerical issues.
            MemoryError: If results contain very large arrays that exceed available memory
                        (relevant for large datasets with high n_points values).

        Side Effects:
            None. This method provides read-only access to fitting results and does not modify
            the internal state of the ELDF object or trigger any recomputation.

        Examples:
            Basic usage after local distribution fitting:
            >>> eldf = ELDF(verbose=True)
            >>> eldf.fit(data)
            >>> results = eldf.results()
            >>> print(f"Local scale parameter: {results['S_opt']:.6f}")
            >>> print(f"Distribution bounds: [{results['LB']:.3f}, {results['UB']:.3f}]")
            >>> print(f"Z0 point: {results['z0']}")
        """
        if not self._fitted:
            raise RuntimeError("Must fit ELDF before getting results.")
        
        return self._get_results()