'''
Marginal Interval Analysis

Take care of end-2-end gnostic process. Primarily work with ELDF.

This module implements the `DataIntervals` class, which provides robust, adaptive, and diagnostic interval estimation for GDF classes such as ELDF, EGDF, QLDF, and QGDF. It estimates meaningful data intervals (such as tolerance and typical intervals) based on the behavior of the GDF's central parameter (Z0) as the data is extended, while enforcing ordering constraints and providing detailed diagnostics.

Author: Nirmal Parmar
Machine Gnostics
'''

import numpy as np
import warnings
import logging
from machinegnostics.magcal.util.logging import get_logger
from machinegnostics.magcal import ELDF, EGDF, DataHomogeneity, DataIntervals, DataCluster, DataMembership

class IntervalAnalysis:
    """
    End-to-End Marginal Interval Analysis for Gnostic Distribution Functions (GDF)

    The `IntervalAnalysis` class provides a robust, automated workflow for estimating meaningful data intervals
    (such as tolerance and typical intervals) using Gnostic Distribution Functions (GDFs) like ELDF and EGDF.
    It is designed for reliability, diagnostics, and adaptive interval estimation in scientific and engineering data analysis.

    This class orchestrates the complete process:
        - Fits an EGDF to the data for global distribution analysis and homogeneity testing.
        - Optionally re-fits for non-homogeneous data and issues warnings if needed.
        - Fits an ELDF for local distribution analysis.
        - Computes robust data intervals using the DataIntervals engine, enforcing ordering constraints.
        - Provides detailed diagnostics, warnings, and error tracking.
        - Offers visualization methods for both the fitted distributions and the estimated intervals.

    Parameters
    ----------
    DLB : float, optional
        Data Lower Bound (absolute minimum possible value for the data).
    DUB : float, optional
        Data Upper Bound (absolute maximum possible value for the data).
    LB : float, optional
        Lower Probable Bound (practical lower limit for the distribution).
    UB : float, optional
        Upper Probable Bound (practical upper limit for the distribution).
    S : float or str, default='auto'
        Scale parameter for the distribution. Use 'auto' for automatic estimation.
    z0_optimize : bool, default=True
        Whether to optimize the central parameter Z0 during fitting.
    tolerance : float, default=1e-9
        Convergence tolerance for optimization.
    data_form : str, default='a'
        Data processing form: 'a' for additive, 'm' for multiplicative.
    n_points : int, default=100
        Number of points for distribution evaluation.
    homogeneous : bool, default=True
        Whether to assume data homogeneity (enables homogeneity testing).
    catch : bool, default=True
        If True, stores warnings/errors and intermediate results.
    weights : np.ndarray, optional
        Prior weights for data points.
    wedf : bool, default=False
        Use Weighted Empirical Distribution Function if True.
    opt_method : str, default='L-BFGS-B'
        Optimization method for parameter estimation.
    verbose : bool, default=False
        Print detailed progress and diagnostics if True.
    max_data_size : int, default=1000
        Maximum data size for smooth GDF generation.
    flush : bool, default=True
        Flush intermediate arrays after fitting to save memory.
    dense_zone_fraction : float, default=0.4
        Fraction of search domain near Z0 for dense interval search.
    dense_points_fraction : float, default=0.7
        Fraction of search points allocated to the dense zone.
    convergence_window : int, default=15
        Window size for convergence detection in interval search.
    convergence_threshold : float, default=1e-6
        Threshold for Z0 convergence in interval search.
    min_search_points : int, default=30
        Minimum search points before checking for convergence.
    boundary_margin_factor : float, default=0.001
        Margin factor to avoid searching exactly at the boundaries.
    extrema_search_tolerance : float, default=1e-6
        Tolerance for detecting extrema in Z0 variation.
    gdf_recompute : bool, default=False
        If True, recompute the GDF for each candidate datum in interval search.
    gnostic_filter : bool, default=False
        If True, apply gnostic clustering to filter outlier Z0 values in interval search.

    Attributes
    ----------
    params : dict
        Stores all warnings, errors, and diagnostic information from the analysis.

    Methods
    -------
    fit(data, plot=False)
        Run the complete interval analysis workflow on the input data.
    results()
        Return a dictionary of estimated interval results and bounds.
    plot(GDF=True, intervals=True)
        Visualize the fitted GDFs and the estimated intervals.

    Usage Example
    -------------
    >>> from machinegnostics.magcal import IntervalAnalysis
    >>> data = np.array([...])
    >>> ia = IntervalAnalysis(verbose=True)
    >>> ia.fit(data, plot=True)
    >>> print(ia.results())
    >>> ia.plot()

    Notes
    -----
    - The class is designed for robust, end-to-end interval estimation and diagnostics.
    - Homogeneity of the data is checked automatically; warnings are issued if violated.
    - For best results, use with ELDF/EGDF and set 'wedf=False' for interval estimation.
    - The class is suitable for scientific, engineering, and reliability applications.
    - All warnings and errors are stored in the `params` attribute for later inspection.

    See Also
    --------
    ELDF, EGDF, DataIntervals

    """
    def __init__(self,
                DLB: float = None,
                DUB: float = None,
                LB: float = None,
                UB: float = None,
                S: str = 'auto',
                z0_optimize: bool = True,
                tolerance: float = 1e-5,
                data_form: str = 'a',
                n_points: int = 100,
                homogeneous: bool = True,
                catch: bool = True,
                weights: np.ndarray = None,
                wedf: bool = False,
                opt_method: str = 'L-BFGS-B',
                verbose: bool = False,
                max_data_size: int = 1000,
                flush: bool = True,
                dense_zone_fraction: float = 0.4,
                dense_points_fraction: float = 0.7,
                convergence_window: int = 15,
                convergence_threshold: float = 0.000001,
                min_search_points: int = 30,
                boundary_margin_factor: float = 0.001,
                extrema_search_tolerance: float = 0.000001,
                gdf_recompute: bool = False,
                gnostic_filter: bool = False,
                cluster_bounds: bool = True,
                membership_bounds: bool = True
                ):
        
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
        self.dense_zone_fraction = dense_zone_fraction
        self.dense_points_fraction = dense_points_fraction
        self.convergence_window = convergence_window
        self.convergence_threshold = convergence_threshold
        self.min_search_points = min_search_points
        self.boundary_margin_factor = boundary_margin_factor
        self.extrema_search_tolerance = extrema_search_tolerance
        self.gdf_recompute = gdf_recompute
        self.gnostic_filter = gnostic_filter
        self.cluster_bounds = cluster_bounds
        self.membership_bounds = membership_bounds
        self._fitted = False

        self.params = {}
        self.params['error'] = []
        self.params['warnings'] = []

        # logger setup
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if verbose else logging.WARNING)
        self.logger.debug(f"{self.__class__.__name__} initialized:")

    def _add_warning(self, warning: str):
        self.params['warnings'].append(warning)
        self.logger.warning(f'Warning: {warning}')
    
    def _add_error(self, error: str):
        self.params['error'].append(error)
        self.logger.error(f'Error: {error}')

    def _input_data_check(self, data: np.ndarray):
        self.logger.info("Checking input data validity.")
        if not isinstance(data, np.ndarray):
            self.logger.error(f'Error: Data must be a numpy array.')
            raise TypeError("Data must be a numpy array.")
        if data.ndim != 1:
            self.logger.error(f'Error: Data must be a 1D array.')
            raise ValueError("Data must be a 1D array.")
        if data.size < 4:
            self.logger.error(f'Error: Data must contain at least 4 elements.')   
            raise ValueError("Data must contain at least 4 elements.")
        # no NaN or Inf values
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            self.logger.error(f'Error: Data contains NaN or Inf values.')
            raise ValueError("Data contains NaN or Inf values.")
        
    def _check_egdf_homogeneity(self, egdf: EGDF):
        self.logger.info("Checking data homogeneity using EGDF.")   
        # check homogeneity
        if self.homogeneous:
            self.dh = DataHomogeneity(gdf=egdf, verbose=self.verbose)
            is_homogeneous = self.dh.fit()
            if not is_homogeneous:
                warning_msg = "Data is not homogeneous. Interval estimation may get affected."
                self._add_warning(warning_msg)
                if self.catch:
                    self.params['warnings'].append(warning_msg)
                    self.params['DataHomogeneity'] = self.dh.params.copy()
                else:
                    self.logger.warning(warning_msg)
        else:
            warning_msg = "Homogeneity check is disabled. Proceeding without checking."
            self._add_warning(warning_msg)
            is_homogeneous = True
            if self.catch:
                self.params['warnings'].append(warning_msg)
            else:
                self.logger.warning(warning_msg)
        return is_homogeneous

    def _get_cluster_bounds(self):
        self.logger.info("Estimating clustering bounds if required.")
        # clustering bounds if required
        if self.cluster_bounds:
            self.logger.info("Cluster bound estimation...")
            self._data_cluster = DataCluster(gdf=self._eldf, verbose=self.verbose, catch=self.catch)
            self.LCB, self.UCB = self._data_cluster.fit()
            if self.catch:
                self.params['DataCluster'] = self._data_cluster.params.copy()
                self.logger.info(f"Updated LCB={self.LCB}, UCB={self.UCB} based on clustering.")
        else:
            self.LCB, self.UCB = None, None
            self.logger.info("Skipping clustering for bound estimation.")

    def _get_membership_bounds(self):
        self.logger.info("Estimating membership bounds if required.")
        # membership bounds if required
        if self.membership_bounds:
            self.logger.info("Estimating data membership bounds...")
            self._data_membership = DataMembership(egdf=self._egdf, verbose=self.verbose, catch=self.catch)
            self.LSB, self.USB = self._data_membership.fit()
            if self.catch:
                self.params['DataMembership'] = self._data_membership.params.copy()
                self.logger.info(f"Updated DLB={self.DLB}, DUB={self.DUB} based on membership.")
        else:
            self.LSB, self.USB = None, None
            self.logger.info("Skipping membership bound estimation.")

    def fit(self, data: np.ndarray, plot: bool = False) -> dict:
        """
        Run the complete marginal interval analysis workflow on the input data. This method takes a 1D numpy array of data and automatically performs all necessary steps to estimate robust data intervals using gnostic distribution functions. It handles data validation, fits the required models, checks for homogeneity, and computes both tolerance and typical intervals with diagnostics. Optionally, it can generate diagnostic plots to help visualize the results. The method returns a dictionary containing the estimated interval bounds and relevant diagnostic information.


        Parameters
        ----------
        data : np.ndarray
            1D numpy array of input data for interval analysis. Must contain at least 4 elements and no NaN/Inf values.
        plot : bool, default=False
            If True, automatically generates diagnostic plots after fitting.

        Returns
        -------
        results : dict
            Dictionary containing estimated interval bounds, tolerance/typical intervals, and diagnostic information.

        Raises
        ------
        TypeError
            If the input data is not a numpy array.
        ValueError
            If the data is not 1D, contains fewer than 4 elements, or contains NaN/Inf values.

        Notes
        -----
        - All warnings and errors encountered during fitting are stored in the `params` attribute.
        - For best results, ensure the data is representative and free of gross outliers.
        - The method sets the `_fitted` attribute to True upon successful completion.

        Example
        -------
        >>> ia = IntervalAnalysis(verbose=True)
        >>> ia.fit(data, plot=True)
        >>> print(ia.results())
        """
        self.logger.info("Starting fit process for IntervalAnalysis.")
        try: 
            # check input data
            self.logger.info("Checking input data...")
            self._input_data_check(data)
            kwargs = {
                'DLB': self.DLB,
                'DUB': self.DUB,
                'LB': self.LB,
                'UB': self.UB,
                'S': self.S,
                'z0_optimize': self.z0_optimize,
                'tolerance': self.tolerance,
                'data_form': self.data_form,
                'n_points': self.n_points,
                'homogeneous': True,
                'catch': self.catch,
                'weights': self.weights,
                'wedf': self.wedf,
                'opt_method': self.opt_method,
                'verbose': self.verbose,
                'max_data_size': self.max_data_size,
                'flush': self.flush
            }
            # estimate EGDF
            self.logger.info("Estimating EGDF...")
            self._egdf = EGDF(**kwargs)
            self._egdf.fit(data)
            if self.catch:
                self.params['EGDF'] = self._egdf.params.copy()

            # check homogeneity
            self.logger.info("Checking data homogeneity...")
            is_homogeneous_1 = self._check_egdf_homogeneity(self._egdf)

            # data must be homogeneous
            if not is_homogeneous_1:
                kwargs_h = {
                'DLB': self.DLB,
                'DUB': self.DUB,
                'LB': self.LB,
                'UB': self.UB,
                'S': self.S,
                'z0_optimize': self.z0_optimize,
                'tolerance': self.tolerance,
                'data_form': self.data_form,
                'n_points': self.n_points,
                'homogeneous': False, # for treating gnostic weight for non-homogeneous data
                'catch': self.catch,
                'weights': self.weights,
                'wedf': self.wedf,
                'opt_method': self.opt_method,
                'verbose': self.verbose,
                'max_data_size': self.max_data_size,
                'flush': self.flush
                }
                self._egdf = EGDF(**kwargs_h)
                self._egdf.fit(data)
                if self.catch:
                    self.params['EGDF_non_homogeneous'] = self._egdf.params.copy()

            # check homogeneity
            self.logger.info("Checking data homogeneity again...")
            is_homogeneous_2 = self._check_egdf_homogeneity(self._egdf)

            # final check on homogeneity, raise warning, that cannot converted to homogeneous, check data
            if not is_homogeneous_2:
                warning_msg = "Data is not homogeneous after re-estimation."
                warning_msg += "Suggested to switch S=1, to improve stability of interval analysis. Advised to process with outliers and re-run OR set S value manually."
                self._add_warning(warning_msg)
                if self.catch:
                    self.params['warnings'].append(warning_msg)
                    self.params['DataHomogeneity'] = self.dh.params.copy()
                else:
                    self.logger.warning(warning_msg)

            # estimate ELDF
            self.logger.info("Estimating ELDF...")
            kwargs_el = {
                'DLB': self.DLB,
                'DUB': self.DUB,
                'LB': self.LB,
                'UB': self.UB,
                'S': self.S, #if (is_homogeneous_1 and self.S == 'auto') else 1, # for non-homogeneous data, set S=1
                'z0_optimize': self.z0_optimize,
                'tolerance': self.tolerance,
                'data_form': self.data_form,
                'n_points': self.n_points,
                'homogeneous': self.homogeneous, # ELDF always assumes homogeneous data
                'catch': self.catch,
                'weights': self.weights,
                'wedf': self.wedf,
                'opt_method': self.opt_method,
                'verbose': self.verbose,
                'max_data_size': self.max_data_size,
                'flush': self.flush
            }
            self._eldf = ELDF(**kwargs_el)
            self._eldf.fit(data)
            if self.catch:
                self.params['ELDF'] = self._eldf.params.copy()

            # get clustering bounds if required
            self.logger.info("Estimating clustering and membership bounds if required.")
            self._get_cluster_bounds()

            # get membership bounds if required
            self.logger.info("Estimating membership bounds if required.")
            if is_homogeneous_2:
                self._get_membership_bounds()
            else:
                self.LSB, self.USB = None, None
                if self.verbose:
                    self._add_warning("Skipping membership bound estimation due to non-homogeneous data.")
                    self.LSB, self.USB = None, None

            # estimate intervals with DataIntervals, minimum compute settings
            self.logger.info("Estimating intervals using DataIntervals engine...")
            di_kwargs = {
                    'gdf': self._eldf,
                    'n_points': self.n_points,
                    'dense_zone_fraction': self.dense_zone_fraction,
                    'dense_points_fraction': self.dense_points_fraction,
                    'convergence_window': self.convergence_window,
                    'convergence_threshold': self.convergence_threshold,
                    'min_search_points': self.min_search_points,
                    'boundary_margin_factor': self.boundary_margin_factor,
                    'extrema_search_tolerance': self.extrema_search_tolerance,
                    'gdf_recompute': self.gdf_recompute,
                    'gnostic_filter': self.gnostic_filter,
                    'catch': self.catch,
                    'verbose': self.verbose,
                    'flush': self.flush
                        }
            self._intv_engine = DataIntervals(**di_kwargs)
            self._intv_engine.fit()

            # z0 and intervals
            self.Z0 = getattr(self._intv_engine, 'Z0', None)
            self.Z0L = getattr(self._intv_engine, 'Z0L', None)
            self.Z0U = getattr(self._intv_engine, 'Z0U', None)
            self.ZL = getattr(self._intv_engine, 'ZL', None)
            self.ZU = getattr(self._intv_engine, 'ZU', None)
            self.LSD = getattr(self._intv_engine, 'LSD', None)
            self.USD = getattr(self._intv_engine, 'USD', None)
            self.DLB = getattr(self._eldf, 'DLB', self.DLB)
            self.DUB = getattr(self._eldf, 'DUB', self.DUB)
            self.LB = getattr(self._eldf, 'LB', self.LB)
            self.UB = getattr(self._eldf, 'UB', self.UB)

            if self.catch:
                self.params['DataIntervals'] = self._intv_engine.params.copy()
            
            # fit status
            self._fitted = True
            self.logger.info("Fit process completed successfully.")
            
            if self.catch:
                self.params['fitted'] = self._fitted

            # if plot is True, generate diagnostic plots
            if plot:
                self.logger.info("Generating diagnostic plots as requested.")
                self._intv_engine.plot()

            return self.results()
        
        except Exception as e:
            self._add_error(str(e))
            raise e

    def results(self) -> dict:
        """
        Return a dictionary of estimated interval results and bounds.

        Returns
        -------
        results : dict
            A dictionary containing the following keys (values may be None if not available):
                - 'LB', 'LSB', 'DLB', 'LCB': Lower bounds (various types, if available)
                - 'ZL': Lower bound of the typical data interval
                - 'Z0L': Lower bound of the tolerance interval (Z0-based)
                - 'Z0': Central value (Z0) of the original GDF
                - 'Z0U': Upper bound of the tolerance interval (Z0-based)
                - 'ZU': Upper bound of the typical data interval
                - 'UCB', 'DUB', 'USB', 'UB': Upper bounds (various types, if available)

        Example
        -------
        >>> intervals = di.results()
        >>> print(intervals['Z0L'], intervals['Z0U'])
        """
        self.logger.info("Compiling results dictionary.")

        results = {
            'LB': float(self.LB) if self.LB is not None else None,
            'LSB': float(self.LSB) if self.LSB is not None else None,
            'DLB': float(self.DLB) if self.DLB is not None else None,
            'LCB': float(self.LCB) if self.LCB is not None else None,
            'LSD': float(self.LSD) if self.LSD is not None else None,
            'ZL': float(self.ZL) if self.ZL is not None else None,
            'Z0L': float(self.Z0L) if self.Z0L is not None else None,
            'Z0': float(self.Z0) if self.Z0 is not None else None,
            'Z0U': float(self.Z0U) if self.Z0U is not None else None,
            'ZU': float(self.ZU) if self.ZU is not None else None,
            'USD': float(self.USD) if self.USD is not None else None,
            'UCB': float(self.UCB) if self.UCB is not None else None,
            'DUB': float(self.DUB) if self.DUB is not None else None,
            'USB': float(self.USB) if self.USB is not None else None,
            'UB': float(self.UB) if self.UB is not None else None
        }
        return results

    def plot(self, GDF: bool = True, intervals: bool = True) -> None:
        """
        Visualize the fitted GDFs (ELDF) and the estimated intervals.

        This method generates diagnostic plots to help interpret the results of the interval analysis:
            - If `GDF` is True, plots the fitted local distribution (ELDF) and its PDF/CDF.
            - If `intervals` is True, plots the Z0 variation and the estimated intervals on the data domain.

        Parameters
        ----------
        GDF : bool, default=True
            If True, plot the fitted ELDF (local distribution function).
        intervals : bool, default=True
            If True, plot the estimated intervals and Z0 variation.

        Returns
        -------
        None

        Notes
        -----
        - Requires matplotlib to be installed.
        - Can be called after `fit()` to visualize the results interactively or in scripts.

        Example
        -------
        >>> ia.plot(GDF=True, intervals=True)
        """
        self.logger.info("Generating plots for fitted GDF and intervals.")

        if hasattr(self, '_intv_engine') and hasattr(self, '_eldf'):
            if GDF:
                self._eldf.plot()
            if intervals:
                self._intv_engine.plot_intervals()
                self._intv_engine.plot()

    def __repr__(self):
        return f"IntervalAnalysis(fitted={self._fitted}, verbose={self.verbose}, results={self.results()})"
    
