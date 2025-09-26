"""
ELDF - Estimating Local Distribution Function.

Author: Nirmal Parmar
Machine Gnostics
"""

import numpy as np
from machinegnostics.magcal.gdf.base_eldf import BaseELDF

class ELDF(BaseELDF):
    """
    Short Description: Estimating Local Distribution Function.

    Detailed Description: The ELDF class estimates and analyzes local distribution functions for given data. It supports both additive and multiplicative data forms, handles bounded and unbounded distributions, and provides advanced Z0 (Gnostic Mean) point estimation. This class is optimized for detailed local analysis and memory-efficient processing.

    Key Features:
        - Automatic parameter estimation with customizable bounds.
        - Advanced Z0 point estimation for maximum PDF location.
        - Support for weighted data points.
        - Multiple data processing forms (additive/multiplicative).
        - Comprehensive visualization capabilities.
        - Robust optimization with multiple solver options.
        - Memory-efficient processing for large datasets.

    Attributes:
        DLB (float): Data Lower Bound - absolute minimum value the data can take.
        DUB (float): Data Upper Bound - absolute maximum value the data can take.
        LB (float): Lower Probable Bound - practical lower limit for the distribution.
        UB (float): Upper Probable Bound - practical upper limit for the distribution.
        S (float or str): Scale parameter for the distribution. Set to 'auto' for automatic estimation.
        varS (bool): Whether to use variable scale parameter during optimization (default: False).
        z0_optimize (bool): Whether to optimize the location parameter Z0 during fitting (default: True).
        tolerance (float): Convergence tolerance for optimization (default: 1e-5).
        data_form (str): Form of the data processing ('a' for additive, 'm' for multiplicative).
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
        plot(plot_smooth=True, plot='eldf', bounds=True, extra_df=True, figsize=(12,8)): Visualize the fitted local distribution.
        results(): Get the fitting results as a dictionary.

    Usage Example:
    
        >>> import numpy as np
        >>> from machinegnostics.magcal import ELDF
        >>> data = np.array([ -13.5, 0, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
        >>> eldf = ELDF()
        >>> eldf.fit(data)
        >>> eldf.plot()
        >>> print(eldf.params)

    Workflow:
        1. Initialize ELDF with desired parameters (no data required).
        2. Call fit(data) to estimate the distribution parameters.
        3. Use plot() to visualize the results.

    Performance Tips:
        - Use data_form='m' for multiplicative/log-normal data.
        - Set appropriate bounds to improve convergence.
        - Use catch=False for large datasets to save memory.
        - Adjust n_points based on visualization needs vs. performance.
        - Use verbose=True to monitor optimization progress.

    Common Use Cases:
        - Peak detection and modal analysis in data distributions.
        - Local density estimation for clustering applications.
        - Risk analysis focusing on critical value identification.
        - Quality control with emphasis on specification limits.
        - Financial modeling with focus on maximum likelihood points.

    Notes:
        - Bounds (DLB, DUB, LB, UB) are optional but can improve estimation accuracy.
        - When S='auto', the scale parameter is automatically estimated from the data.
        - The weights array must have the same length as the data array.
        - Setting catch=False can save memory for large datasets but prevents access to intermediate results or detailed plots.

    Raises:
        ValueError: If data array is empty or contains invalid values.
        ValueError: If weights array length doesn't match data array length.
        ValueError: If bounds are specified incorrectly (e.g., LB > UB).
        ValueError: If invalid parameters are provided (negative tolerance, invalid data_form, etc.).
        RuntimeError: If the fitting process fails to converge.
        OptimizationError: If the optimization algorithm encounters numerical issues.
    """

    def __init__(self,
                 DLB: float = None,
                 DUB: float = None,
                 LB: float = None,
                 UB: float = None,
                 S = 'auto',
                 varS: bool = False,
                 z0_optimize: bool = True,
                 tolerance: float = 1e-9,
                 data_form: str = 'a',
                 n_points: int = 1000,
                 homogeneous: bool = True,
                 catch: bool = True,
                 weights: np.ndarray = None,
                 wedf: bool = False,
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
            varS (bool, optional): Whether to allow variable scale parameter during optimization to handle heteroscedasticity.
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
        
            >>> import numpy as np
            >>> from machinegnostics.magcal import ELDF
            >>> data = np.array([ -13.5, 0, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
            >>> eldf = ELDF()
            >>> eldf.fit(data)
            >>> eldf.plot()
            >>> print(eldf.params)
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
        Short Description: Fit the Estimating Local Distribution Function to the provided data.

        Detailed Description: This method performs the core estimation process for the ELDF. It validates and preprocesses the data, sets up optimization constraints, runs numerical optimization, and calculates the final ELDF and PDF with optimized parameters. The ELDF provides detailed local analysis, including Z0 estimation (location of maximum PDF).

        Parameters:
            data (np.ndarray): Input data array for distribution estimation. Must be a 1D numpy array.
            plot (bool, optional): Whether to automatically plot the fitted distribution after fitting. Default is False.

        Returns:
            None: The fitted parameters are stored in the `params` attribute.

        Raises:
            RuntimeError: If the optimization process fails to converge.
            ValueError: If the data array is empty, contains only NaN values, or has invalid dimensions.
            ValueError: If weights array is provided but has a different length than the data array.
            OptimizationError: If the underlying optimization algorithm encounters numerical issues.
            ConvergenceError: If the algorithm cannot find a suitable solution.

        Usage Example:

            >>> eldf = ELDF()
            >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            >>> eldf.fit(data)
            >>> print("Fitting completed")
            >>> print(f"Fitted parameters: {eldf.params}")
            >>> eldf.fit(data, plot=True)
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

    def plot(self, plot_smooth: bool = True, plot: str = 'both', bounds: bool = True, extra_df: bool = True, figsize: tuple = (12, 8)):
        """
        Short Description: Visualize the fitted Estimating Local Distribution Function and related plots.

        Detailed Description: This method generates visualizations of the fitted local distribution function, including the main ELDF curve, probability density function (PDF), and optional additional distribution functions. It provides insights into the quality of the fit and the characteristics of the underlying local distribution.

        Parameters:
            plot_smooth (bool, optional): Whether to plot a smooth interpolated curve for the distribution function. Default is True.
            plot (str, optional): Type of plot to generate. Default is 'both'. Options include:
                - 'eldf': Local Distribution Function (main curve).
                - 'pdf': Probability Density Function.
                - 'both': Both ELDF and PDF in the same plot.
            bounds (bool, optional): Whether to display bound lines on the plot. Default is True.
            extra_df (bool, optional): Whether to include additional distribution functions in the plot for comparison. Default is True.
            figsize (tuple, optional): Figure size as (width, height) in inches. Default is (12, 8).

        Returns:
            None: Displays the plot.

        Raises:
            RuntimeError: If fit() has not been called before plotting.
            ValueError: If an invalid plot type is specified.
            ImportError: If matplotlib is not available for plotting.
            PlottingError: If there are issues with the plot generation process.
            MemoryError: If plotting large datasets exceeds available memory.

        Usage Example:

            >>> eldf.plot()
            >>> eldf.plot(plot='pdf', bounds=True)
            >>> eldf.plot(plot='both', bounds=True, extra_df=True, figsize=(16, 10))
        """
        self._plot(plot_smooth=plot_smooth, plot=plot, bounds=bounds, extra_df=extra_df, figsize=figsize)

    def results(self) -> dict:
        """
        Short Description: Retrieve the fitted parameters and comprehensive results from the ELDF fitting process.

        Detailed Description: This method provides access to all key results obtained after fitting the Estimating Local Distribution Function (ELDF) to the data. It returns a dictionary containing fitted parameters, local distribution characteristics, optimization results, and diagnostic information for complete distribution analysis.

        Returns:
            dict: Fitted parameters and results.

        Raises:
            RuntimeError: If fit() has not been called before accessing results.
            AttributeError: If internal result structure is missing or corrupted due to fitting failure.
            KeyError: If expected result keys are unavailable.
            ValueError: If internal state is inconsistent for result retrieval.
            MemoryError: If results contain very large arrays that exceed available memory.

        Usage Example:
        
            >>> eldf = ELDF(verbose=True)
            >>> eldf.fit(data)
            >>> results = eldf.results()
            >>> print(f"Local scale parameter: {results['S_opt']:.6f}")
            >>> print(f"Distribution bounds: [{results['LB']:.3f}, {results['UB']:.3f}]")
        """
        if not self._fitted:
            raise RuntimeError("Must fit ELDF before getting results.")
        return self._get_results()