"""
EGDF - Estimating Global Distribution Function.

Author: Nirmal Parmar
Machine Gnostics
"""

import numpy as np
from machinegnostics.magcal.gdf.base_egdf import BaseEGDF

class EGDF(BaseEGDF):
    """
    Short Description: Estimating Global Distribution Function.

    Detailed Description: The EGDF class estimates and analyzes global distribution functions for given data. It supports both additive and multiplicative data forms, handles bounded and unbounded distributions, and provides automatic parameter estimation and visualization options. This class is optimized for robust optimization and memory-efficient processing of large datasets.

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
        z0_optimize (bool): Whether to optimize the location parameter z0 during fitting (default: True).
        data_form (str): Form of the data processing ('a' for additive, 'm' for multiplicative).
        n_points (int): Number of points to generate in the distribution function (default: 500).
        catch (bool): Whether to store intermediate calculated values (default: True).
        weights (np.ndarray): Prior weights for data points. If None, uniform weights are used.
        wedf (bool): Whether to use Weighted Empirical Distribution Function (default: False).
        opt_method (str): Optimization method for parameter estimation (default: 'L-BFGS-B').
        tolerance (float): Convergence tolerance for optimization (default: 1e-9).
        verbose (bool): Whether to print detailed progress information (default: False).
        params (dict): Dictionary storing fitted parameters and results after fitting.
        homogeneous (bool): To indicate data homogeneity (default: True).
        max_data_size (int): Maximum data size for smooth EGDF generation (default: 1000).
        flush (bool): Whether to flush large arrays (default: True).

    Methods:
        fit(data): Fit the Estimating Global Distribution Function to the data.
        plot(plot_smooth=True, plot='gdf', bounds=False, extra_df=True, figsize=(12,8)): Visualize the fitted distribution.
        results(): Get the fitting results as a dictionary.

    Usage Example:
    
        >>> import numpy as np
        >>> from machinegnostics.magcal import EGDF
        >>> data = np.array([ -13.5, 0, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
        >>> egdf = EGDF()
        >>> egdf.fit(data)
        >>> egdf.plot()
        >>> print(egdf.params)

    Workflow:
        1. Initialize EGDF with desired parameters (no data required).
        2. Call fit(data) to estimate the distribution parameters.
        3. Use plot() to visualize the results.

    Performance Tips:
        - Use data_form='m' for multiplicative/log-normal data.
        - Set appropriate bounds to improve convergence.
        - Use catch=False for large datasets to save memory.
        - Adjust n_points based on visualization needs vs. performance.
        - Use verbose=True to monitor optimization progress.

    Common Use Cases:
        - Risk analysis and reliability engineering.
        - Quality control and process optimization.
        - Financial modeling and market analysis.
        - Environmental data analysis.
        - Biostatistics and epidemiological studies.

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
                z0_optimize: bool = True,
                tolerance: float = 1e-9,
                data_form: str = 'a',
                n_points: int = 500,
                homogeneous: bool = True,
                catch: bool = True,
                weights: np.ndarray = None,
                wedf: bool = False,
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
            S (float, optional): Scale parameter for the distribution. If 'auto' (default),
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
                                 calculations. Default is False. When True, incorporates weights
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
        
            >>> import numpy as np
            >>> from machinegnostics.magcal import EGDF
            >>> data = np.array([ -13.5, 0, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
            >>> egdf = EGDF()
            >>> egdf.fit(data)
            >>> egdf.plot()
            >>> print(egdf.params)
        
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
        Short Description: Fit the Estimating Global Distribution Function to the provided data.

        Detailed Description: This method performs the core estimation process for the EGDF. It validates and preprocesses the data, sets up optimization constraints, runs numerical optimization, and calculates the final EGDF and PDF with optimized parameters. The EGDF provides a unique global representation of the data distribution.

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

            >>> egdf = EGDF()
            >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            >>> egdf.fit(data)
            >>> print("Fitting completed")
            >>> print(f"Fitted parameters: {egdf.params}")
            >>> egdf.fit(data, plot=True)
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

    def plot(self, plot_smooth: bool = True, plot: str = 'both', bounds: bool = False, extra_df: bool = True, figsize: tuple = (12, 8)):
        """
        Short Description: Visualize the fitted Estimating Global Distribution Function and related plots.

        Detailed Description: This method generates visualizations of the fitted global distribution function, including the main EGDF curve, probability density function (PDF), and optional additional distribution functions. It provides insights into the quality of the fit and the characteristics of the underlying distribution.

        Parameters:
            plot_smooth (bool, optional): Whether to plot a smooth interpolated curve for the distribution function. Default is True.
            plot (str, optional): Type of plot to generate. Default is 'both'. Options include:
                - 'gdf': Global Distribution Function (main curve).
                - 'pdf': Probability Density Function.
                - 'both': Both EGDF and PDF in the same plot.
            bounds (bool, optional): Whether to display bound lines on the plot. Default is False.
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

            >>> egdf.plot()
            >>> egdf.plot(plot='pdf', bounds=True)
            >>> egdf.plot(plot='both', bounds=True, extra_df=True, figsize=(16, 10))
        """
        self._plot(plot_smooth=plot_smooth, plot=plot, bounds=bounds, extra_df=extra_df, figsize=figsize)

    def results(self) -> dict:
        """
        Short Description: Retrieve the fitted parameters and comprehensive results from the EGDF fitting process.

        Detailed Description: This method provides access to all key results obtained after fitting the Estimating Global Distribution Function (EGDF) to the data. It returns a dictionary containing fitted parameters, global distribution characteristics, optimization results, and diagnostic information for complete distribution analysis.

        Returns:
            dict: Fitted parameters and results.

        Raises:
            RuntimeError: If fit() has not been called before accessing results.
            AttributeError: If internal result structure is missing or corrupted due to fitting failure.
            KeyError: If expected result keys are unavailable.
            ValueError: If internal state is inconsistent for result retrieval.
            MemoryError: If results contain very large arrays that exceed available memory.

        Usage Example:
        
            >>> egdf = EGDF(verbose=True)
            >>> egdf.fit(data)
            >>> results = egdf.results()
            >>> print(f"Global scale parameter: {results['S_opt']:.6f}")
            >>> print(f"Distribution bounds: [{results['LB']:.3f}, {results['UB']:.3f}]")
        """
        if not self._fitted:
            raise RuntimeError("Must fit EGDF before getting results.")
        return self._get_results()