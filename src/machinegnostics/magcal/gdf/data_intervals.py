'''
DataIntervals

Interval Analysis Engine

Author: Nirmal Parmar
Machine Gnostics
'''

import logging
import numpy as np
from typing import Optional, Union, Dict
from scipy.signal import savgol_filter, find_peaks
from machinegnostics.magcal import ELDF, EGDF, QLDF, QGDF, DataCluster
from machinegnostics.metrics.std import std
from machinegnostics.magcal.util.logging import get_logger

class DataIntervals:
    """
    Robust Interval Estimation Engine for GDF Classes

    The `DataIntervals` class provides robust, adaptive, and diagnostic interval estimation for 
    Gnostics Distribution Function (GDF) classes such as ELDF, EGDF, QLDF, and QGDF. 
    It is designed to estimate meaningful data intervals (such as tolerance and typical intervals) 
    based on the behavior of the GDF's central parameter (Z0) as the data is extended, 
    while enforcing ordering constraints and providing detailed diagnostics.

    Key Features:
    -------------
    - **Adaptive Search:** 
      Efficiently scans the data domain with a dense search near the central value (Z0) and 
      sparser search near the boundaries, balancing computational cost and accuracy.
    - **Robustness:** 
      Supports optional recomputation of the GDF for each candidate datum, with optional 
      gnostic filtering (clustering) to enhance robustness against outliers and noise.
    - **Diagnostics:** 
      Provides warnings and errors for suboptimal settings, ordering violations, and 
      insufficient data, and stores detailed parameters and results for further analysis.
    - **Ordering Constraint:** 
      Ensures that the estimated intervals satisfy the natural ordering: 
      ZL < Z0L < Z0 < Z0U < ZU, where ZL/ZU are typical data interval bounds and 
      Z0L/Z0U are tolerance interval bounds.
    - **Visualization:** 
      Offers plotting methods to visualize the Z0 variation, estimated intervals, 
      and data coverage.

    Parameters
    ----------
    gdf : ELDF, EGDF, QLDF, or QGDF
        The fitted GDF (Gnostics Distribution Function) object to analyze.
    n_points : int, default=100
        Number of search points for interval estimation (minimum 50).
    dense_zone_fraction : float, default=0.4
        Fraction of the search domain near Z0 to search densely (range: 0.1 to 0.8).
    dense_points_fraction : float, default=0.7
        Fraction of points allocated to the dense zone (range: 0.5 to 0.9).
    convergence_window : int, default=15
        Number of points in the moving window for convergence detection.
    convergence_threshold : float, default=1e-6
        Threshold for standard deviation of Z0 in the convergence window.
    min_search_points : int, default=30
        Minimum number of search points before checking for convergence.
    boundary_margin_factor : float, default=0.001
        Margin factor to avoid searching exactly at the boundaries.
    extrema_search_tolerance : float, default=1e-6
        Tolerance for detecting extrema in Z0 variation.
    gdf_recompute : bool, default=False
        If True, recompute the GDF for each candidate datum.
    gnostic_filter : bool, default=False
        If True, apply gnostic clustering to filter outlier Z0 values.
    catch : bool, default=True
        If True, catch and store warnings/errors internally.
    verbose : bool, default=False
        If True, print detailed progress and diagnostics.
    flush : bool, default=False
        If True, flush memory after fitting to save resources.

    Attributes
    ----------
    ZL : float
        Lower bound of the typical data interval.
    Z0L : float
        Lower bound of the tolerance interval (Z0-based).
    Z0 : float
        Central value (Z0) of the original GDF.
    Z0U : float
        Upper bound of the tolerance interval (Z0-based).
    ZU : float
        Upper bound of the typical data interval.
    tolerance_interval : float
        Width of the tolerance interval (Z0U - Z0L).
    typical_data_interval : float
        Width of the typical data interval (ZU - ZL).
    ordering_valid : bool
        Whether the ordering constraint (ZL < Z0L < Z0 < Z0U < ZU) is satisfied.
    params : dict
        Dictionary of parameters, warnings, errors, and results.
    search_results : dict
        Raw search results for datum values and corresponding Z0s.

    Methods
    -------
    fit(plot=False)
        Run the interval estimation process. Optionally plot results.
    results() -> dict
        Return a dictionary of interval results and bounds.
    plot_intervals(figsize=(12, 8))
        Plot the Z0 variation and estimated intervals.
    plot(figsize=(12, 8))
        Plot the GDF, PDF, and intervals on the data domain.

    Usage Example
    -------------
    >>> eld = ELDF()
    >>> eld.fit(data)
    >>> di = DataIntervals(eld, n_points=200, gdf_recompute=True, gnostic_filter=True, verbose=True)
    >>> di.fit(plot=True)
    >>> print(di.results())
    >>> di.plot_intervals()
    >>> di.plot()

    Notes
    -----
    - For best results, use with ELDF or QLDF and set 'wedf=False' in the GDF.
    - Increasing 'n_points' improves accuracy but increases computation time.
    - Enable 'gdf_recompute' and 'gnostic_filter' for maximum robustness, especially with noisy data.
    - The class is designed for research and diagnostic use; adjust parameters for your data and application.
    """
    def __init__(self, gdf: Union[ELDF, EGDF, QLDF, QGDF],
                 n_points: int = 100,
                 dense_zone_fraction: float = 0.4,
                 dense_points_fraction: float = 0.7,
                 convergence_window: int = 15,
                 convergence_threshold: float = 1e-6,
                 min_search_points: int = 30,
                 boundary_margin_factor: float = 0.001,
                 extrema_search_tolerance: float = 1e-6,
                 gdf_recompute: bool = False,
                 gnostic_filter: bool = False,
                 catch: bool = True,
                 verbose: bool = False,
                 flush: bool = False):
        self.gdf = gdf
        self.n_points = max(n_points, 50)
        self.dense_zone_fraction = np.clip(dense_zone_fraction, 0.1, 0.8)
        self.dense_points_fraction = np.clip(dense_points_fraction, 0.5, 0.9)
        self.convergence_window = max(convergence_window, 5)
        self.convergence_threshold = convergence_threshold
        self.min_search_points = max(min_search_points, 10)
        self.boundary_margin_factor = max(boundary_margin_factor, 1e-6)
        self.extrema_search_tolerance = extrema_search_tolerance
        self.gdf_recompute = gdf_recompute
        self.gnostic_filter = gnostic_filter
        self.catch = catch
        self.verbose = verbose
        self.flush = flush
        self.params: Dict = {}
        self.params['errors'] = []
        self.params['warnings'] = []
        self.search_results = {'datum': [], 'z0': [], 'success': []}

        # logger setup
        self.logger = get_logger(self.__class__.__name__, level=logging.DEBUG if self.verbose else logging.WARNING)
        self.logger.debug(f"{self.__class__.__name__} initialized:")

        # checks
        self._extract_gdf_data()
        self._reset_results()
        self._store_init_params()

        # validation
        # n_points should not less then 50 or more then 10000 else it can be computationally expensive. It balances efficiency and accuracy.
        if self.n_points < 50 or self.n_points > 10000:
            msg =  f"n_points={self.n_points} is out of recommended range [50, 10000]. Consider adjusting for efficiency and accuracy."
            self._add_warning(msg)

        # if gdf_recompute = True, it is recommended to use gnostic_filter = True to enhance robustness.
        if self.gdf_recompute and not self.gnostic_filter:
            msg = "Using gdf_recompute=True without gnostic_filter=True may reduce robustness. Consider enabling gnostic_filter if needed."
            self._add_warning(msg)

    def _add_warning(self, message: str):
        self.params['warnings'].append(message)
        self.logger.warning(f"Warning: {message}")
        if self.catch:
            self.params['warnings'].append(message)
    
    def _add_error(self, message: str):
        self.params['errors'].append(message)
        self.logger.error(f"Error: {message}")
        if self.catch:
            self.params['errors'].append(message)

    def _extract_gdf_data(self):
        self.logger.info(f"Extracting GDF data...")
        try:
            gdf = self.gdf
            self.data = np.array(gdf.data)
            self.Z0 = float(gdf.z0)
            self.LB = float(gdf.LB)
            self.UB = float(gdf.UB)
            self.S = getattr(gdf, 'S', 'auto')
            self.S_opt = getattr(gdf, 'S_opt', None)
            self.wedf = getattr(gdf, 'wedf', False)
            self.n_points_gdf = getattr(gdf, 'n_points', self.n_points)
            self.opt_method = getattr(gdf, 'opt_method', 'L-BFGS-B')
            self.homogeneous = getattr(gdf, 'homogeneous', True)
            self.is_homogeneous = getattr(gdf, 'is_homogeneous', True)
            self.z0_optimize = getattr(gdf, 'z0_optimize', True)
            self.max_data_size = getattr(gdf, 'max_data_size', 1000)
            self.tolerance = getattr(gdf, 'tolerance', 1e-5)
            self.DLB = getattr(gdf, 'DLB', None)
            self.DUB = getattr(gdf, 'DUB', None)
            self.LSB = getattr(gdf, 'LSB', None)
            self.USB = getattr(gdf, 'USB', None)
            self.LCB = getattr(gdf, 'LCB', None)
            self.UCB = getattr(gdf, 'UCB', None)
            self.RRE = self.gdf.params.get('RRE', None)
            self.residual_entropy = self.gdf.params.get('residual_entropy', None)
            self.gdf_name = type(gdf).__name__
            if self.catch:
                self.params['gdf_type'] = self.gdf_name
                self.params['data_size'] = len(self.data)
                self.params['LB'] = self.LB
                self.params['UB'] = self.UB
                self.params['Z0'] = self.Z0
                self.params['S'] = self.S
                self.params['S_opt'] = self.S_opt
                self.params['wedf'] = self.wedf
                self.params['opt_method'] = self.opt_method
                self.params['is_homogeneous'] = self.is_homogeneous
                self.params['data_range'] = [float(np.min(self.data)), float(np.max(self.data))]
                self.params['RRE'] = self.RRE
                self.params['residual_entropy'] = self.residual_entropy

            self.logger.debug(f"Initialized with {self.params['gdf_type']} | Data size: {self.params['data_size']} | Z0: {self.Z0:.6f}")

        except Exception as e:
            self._add_error(f"Failed to extract GDF data: {e}")
            return
        
    def _argument_validation(self):
        self.logger.info("Validating arguments and settings...")
        # Check GDF type suitability
        if self.gdf_name not in ['ELDF', 'QLDF']:
            msg = "Interval Analysis is optimized for ELDF and QLDF. Results may be less robust for other types."
            self._add_warning(msg)
    
        # Check wedf setting
        if getattr(self.gdf, 'wedf', False):
            msg = "Interval Analysis works best with KSDF. Consider setting 'wedf=False' for optimal results."
            self._add_warning(msg)
    
        # Check n_points for computational efficiency
        if self.n_points > 1000:
            msg = (f"Current n_points = {self.n_points} is very high and may cause excessive computation time. "
                   "Consider reducing n_points for efficiency.")
            self._add_warning(msg)

    def _store_init_params(self):
        if self.catch:
            self.params.update({
                'n_points': self.n_points,
                'dense_zone_fraction': self.dense_zone_fraction,
                'dense_points_fraction': self.dense_points_fraction,
                'convergence_window': self.convergence_window,
                'convergence_threshold': self.convergence_threshold,
                'min_search_points': self.min_search_points,
                'boundary_margin_factor': self.boundary_margin_factor,
                'extrema_search_tolerance': self.extrema_search_tolerance,
                'verbose': self.verbose,
                'flush': self.flush
            })
        self.logger.info("Initial parameters stored.")

    def _reset_results(self):
        self.ZL = None
        self.Z0L = None
        self.ZU = None
        self.Z0U = None
        self.tolerance_interval = None
        self.typical_data_interval = None
        self.ordering_valid = None

    def fit(self, plot: bool = False):
        """
        Run the interval estimation process for the fitted GDF.

        This method performs adaptive interval scanning by extending the data with candidate values,
        recomputing the GDF (if enabled), and tracking the variation of the central parameter Z0.
        It then extracts the typical data interval and tolerance interval, checks the ordering constraint,
        and updates the internal results and diagnostics.

        Parameters
        ----------
        plot : bool, optional (default=False)
            If True, automatically plot the interval analysis results after fitting.

        Raises
        ------
        Exception
            If the fitting process fails due to invalid arguments or internal errors.

        Notes
        -----
        - For best results, ensure the GDF is already fitted to data before calling this method.
        - The method updates the object's attributes with the estimated intervals and stores
          diagnostics in the `params` attribute.
        - If `flush=True` was set at initialization, intermediate data is cleared after fitting.
        """
        self.logger.info("Starting fit process for DataIntervals...")
        import time
        start_time = time.time()
        try:
            self._argument_validation()

            self.logger.info("Fit process started.")
            self._reset_results()

            # Scan intervals and extract boundaries
            self._scan_intervals()
            self._extract_intervals_with_ordering()
    
            # Check ordering constraint
            if not self.ordering_valid:
                msg = ("Interval ordering constraint violated. "
                       "Try setting 'wedf=False', or setting 'gnostic_filter=True', or increasing 'n_points', or adjusting thresholds for sensitivity.")
                self._add_warning(msg)

            # std interval
            self.LSD, self.USD= std(self.data, S=self.S_opt, z0_optimize=self.z0_optimize, data_form=self.gdf.data_form, tolerance=self.tolerance)
            # Update parameters and optionally plot
            self._update_params()
            if plot:
                self.logger.info("Plotting interval analysis results...")
                self.plot()
                self.plot_intervals()          
    
            # Optionally flush memory
            if self.flush:
                self.logger.info("Flushing memory...")
                self._flush_memory()
    
            elapsed = time.time() - start_time
            self.logger.info(f"Fit process completed in {elapsed:.2f} seconds.")
            self.logger.info(f"Ordering valid: {self.ordering_valid}")
            self.logger.info(f"Tolerance interval: [{self.Z0L:.4f}, {self.Z0U:.4f}]")
            self.logger.info(f"Typical data interval: [{self.ZL:.4f}, {self.ZU:.4f}]")
        except Exception as e:
            err_msg = f"Fit failed: {e}"
            self._add_error(err_msg)
            raise

    def _scan_intervals(self):
        self.logger.info("Scanning intervals by extending data...")
        try:
            self.logger.info("Scanning intervals...")

            # Scan lower direction (Z0 -> LB)
            lower_points = self._generate_search_points('lower')
            self.logger.info(f"  Starting lower scan: {len(lower_points)} points from Z0 → LB")
            for i, datum in enumerate(lower_points, 1):
                z0_val = self._compute_z0_with_extended_datum(datum)
                self.search_results['datum'].append(datum)
                self.search_results['z0'].append(z0_val)
                self.search_results['success'].append(True)
                if self.verbose and i % (self.n_points/10) == 0:
                    self.logger.info(f"    Lower scan [{i}/{len(lower_points)}]: Datum={datum:.4f}, Z0={z0_val:.6f}")
                if self._check_convergence():
                    if self.verbose:
                        self.logger.info(f"  Early stopping in lower scan at datum={datum:.4f}")
                    return  # stop scanning entirely if convergence is reached

            # Scan upper direction (Z0 -> UB)
            upper_points = self._generate_search_points('upper')
            self.logger.info(f"  Starting upper scan: {len(upper_points)} points from Z0 → UB")
            for i, datum in enumerate(upper_points, 1):
                z0_val = self._compute_z0_with_extended_datum(datum)
                self.search_results['datum'].append(datum)
                self.search_results['z0'].append(z0_val)
                self.search_results['success'].append(True)
                if self.verbose and i % 50 == 0:
                    self.logger.info(f"    Upper scan [{i}/{len(upper_points)}]: Datum={datum:.4f}, Z0={z0_val:.6f}")
                if self._check_convergence():
                    if self.verbose:
                        self.logger.info(f"  Early stopping in upper scan at datum={datum:.4f}")
                    return

        except Exception as e:
            self._add_error(f"Scanning intervals failed: {e}")
            return


    def _generate_search_points(self, direction: str) -> np.ndarray:
        self.logger.debug(f"Generating search points in {direction} direction...")
        # Dense zone near Z0, sparse toward LB/UB
        if direction == 'lower':
            start, end = self.Z0, self.LB + self.boundary_margin_factor * (self.UB - self.LB)
        else:
            start, end = self.Z0, self.UB - self.boundary_margin_factor * (self.UB - self.LB)
        dense_n = int(self.n_points * self.dense_points_fraction)
        sparse_n = self.n_points - dense_n
        dense_zone = self.dense_zone_fraction * abs(self.Z0 - end)
        if direction == 'lower':
            dense_end = self.Z0 - dense_zone
            dense_points = np.linspace(self.Z0, dense_end, dense_n)
            sparse_points = np.linspace(dense_end, end, sparse_n)
        else:
            dense_end = self.Z0 + dense_zone
            dense_points = np.linspace(self.Z0, dense_end, dense_n)
            sparse_points = np.linspace(dense_end, end, sparse_n)
        return np.unique(np.concatenate([dense_points, sparse_points]))

    def _compute_z0_with_extended_datum(self, datum: float) -> float:
        self.logger.info(f"Computing Z0 with extended datum: {datum:.4f}")
        # Extend data and fit new GDF, return z0
        extended_data = np.append(self.data, datum)
        gdf_type = type(self.gdf)
        if self.gdf_recompute:
            kwargs = {
                'verbose': False,
                'flush': True,
                'opt_method': self.opt_method,
                'n_points': self.n_points_gdf,
                'wedf': self.wedf,
                'homogeneous': self.homogeneous,
                'z0_optimize': self.z0_optimize,
                'max_data_size': self.max_data_size,
                'tolerance': self.tolerance,
            }
        else:
            kwargs = {
                    'LB': self.LB,
                    'UB': self.UB,
                    'S': self.S,
                    'verbose': False,
                    'flush': True,
                    'opt_method': self.opt_method,
                    'n_points': self.n_points_gdf,
                    'wedf': self.wedf,
                    'homogeneous': self.homogeneous,
                    'z0_optimize': self.z0_optimize,
                    'max_data_size': self.max_data_size,
                    'tolerance': self.tolerance,
                }
        gdf_new = gdf_type(**kwargs)
        gdf_new.fit(data=extended_data, plot=False)
        return float(gdf_new.z0)

    def _check_convergence(self) -> bool:
        self.logger.info("Checking convergence of Z0...")
        z0s = np.array(self.search_results['z0'])
        if len(z0s) < self.convergence_window + self.min_search_points:
            return False
        window = z0s[-self.convergence_window:]
        if np.std(window) < self.convergence_threshold:
            return True
        return False

    def _get_z0s_main_cluster(self, z0s: np.ndarray, datums: np.ndarray) -> np.ndarray:
        self.logger.info("Extracting main Z0 cluster...")
        try:
            # 4 less data points - skip clustering
            if len(z0s) <= 4 or len(datums) < 4:
                self._add_warning("Insufficient data points for clustering. Returning all values.")
                return z0s, datums

            # Fit ELDF to z0s for clustering
            self.logger.info("Fitting ELDF for clustering...")
            eldf_cluster = ELDF(catch=False, wedf=False, verbose=False)
            eldf_cluster.fit(z0s)
            # Cluster boundaries
            self.logger.info("Fitting DataCluster to identify main cluster...")
            cluster = DataCluster(gdf=eldf_cluster, verbose=self.verbose)
            clb, cub = cluster.fit()

            # z0s within cluster boundaries
            in_cluster_mask = (z0s >= clb) & (z0s <= cub)
            if not np.any(in_cluster_mask):
                self._add_warning("No Z0 values found within cluster boundaries. Returning all values.")
                return z0s, datums

            z0s_main = z0s[in_cluster_mask]
            datums_main = datums[in_cluster_mask]
            return z0s_main, datums_main
    
        except Exception as e:
            self._add_warning(f"Cluster-based Z0 extraction failed: {e}. Using all Z0 values.")
            return np.array(self.search_results['z0']), np.array(self.search_results['datum'])

    def _extract_intervals_with_ordering(self):
        self.logger.info("Extracting intervals with ordering constraint...")

        datums = np.array(self.search_results['datum'])
        z0s = np.array(self.search_results['z0'])

        if self.gnostic_filter:
            self.logger.info("Applying gnostic filtering to Z0 values...")
            # MG cluster
            z0s, datums = self._get_z0s_main_cluster(z0s, datums)

        # Smoothing
        if len(z0s) > 11:
            z0s_smooth = savgol_filter(z0s, 11, 3)
        else:
            z0s_smooth = z0s

        # clean dict
        self.search_results_clean = {
            'datum': datums,
            'z0': z0s_smooth
            }

        # Window
        data_mean = np.mean(self.data)
        data_std = np.std(self.data)
        window_mask = (datums >= data_mean - 2 * data_std) & (datums <= data_mean + 2 * data_std)
        datums_win = datums[window_mask]
        z0s_win = z0s_smooth[window_mask]
        if len(z0s_win) == 0:
            datums_win = datums
            z0s_win = z0s_smooth
    
        # Find local minima/maxima with prominence
        min_peaks, _ = find_peaks(-z0s_win, prominence=0.1)
        max_peaks, _ = find_peaks(z0s_win, prominence=0.1)
        # Fallback to global min/max if no peaks found
        if len(min_peaks) > 0:
            min_idx = min_peaks[np.argmin(z0s_win[min_peaks])]
        else:
            min_idx = np.argmin(z0s_win)
        if len(max_peaks) > 0:
            max_idx = max_peaks[np.argmax(z0s_win[max_peaks])]
        else:
            max_idx = np.argmax(z0s_win)
        zl, z0l = datums_win[min_idx], z0s_win[min_idx]
        zu, z0u = datums_win[max_idx], z0s_win[max_idx]
        ordering_valid = (zl < z0l < self.Z0 < z0u < zu)
        if ordering_valid:
            self.ZL, self.Z0L, self.ZU, self.Z0U = zl, z0l, zu, z0u
            self.ordering_valid = True
        else:
            self._find_valid_extrema_with_ordering(datums_win, z0s_win)

        # is still invalid? then replace incorrect bounds with z0
        if not self.ordering_valid:
            if self.Z0 < self.Z0L:
                self.Z0L = self.Z0
            if self.Z0 > self.Z0U:
                self.Z0U = self.Z0
            if self.ZL > self.Z0L:
                self.ZL = self.Z0L
            if self.ZU < self.Z0U:
                self.ZU = self.Z0U

            self.logger.info("Adjusted bounds to enforce ordering constraint.")
        self.tolerance_interval = self.Z0U - self.Z0L
        self.typical_data_interval = self.ZU - self.ZL

    def _find_valid_extrema_with_ordering(self, datums, z0s):
        self.logger.info("Searching for valid extrema combinations to satisfy ordering constraint...")
        # Try combinations to satisfy ordering constraint
        lower_mask = datums < self.Z0
        upper_mask = datums > self.Z0
        lower_datum = datums[lower_mask]
        lower_z0 = z0s[lower_mask]
        upper_datum = datums[upper_mask]
        upper_z0 = z0s[upper_mask]
        n_candidates = min(5, len(lower_datum), len(upper_datum))
        found = False

        self.logger.info(f"Found {n_candidates} candidate pairs for extrema.")
        for i in range(n_candidates):
            zl, z0l = lower_datum[i], lower_z0[i]
            zu, z0u = upper_datum[-(i+1)], upper_z0[-(i+1)]
            if zl < z0l < self.Z0 < z0u < zu:
                self.ZL, self.Z0L, self.ZU, self.Z0U = zl, z0l, zu, z0u
                self.ordering_valid = True
                found = True
                break

        self.logger.info(f"Valid extrema found: {found}")
        if not found:
            # Fallback: use initial extrema
            min_idx = np.argmin(z0s)
            max_idx = np.argmax(z0s)
            self.ZL, self.Z0L, self.ZU, self.Z0U = datums[min_idx], z0s[min_idx], datums[max_idx], z0s[max_idx]
            self.ordering_valid = False

        self.logger.info(f"Ordering constraint {'satisfied' if self.ordering_valid else 'NOT satisfied'}.")

    def _update_params(self):
        self.logger.info("Updating parameters with results...")
        self.params.update({
            'LB': self.LB,
            'LSB': self.LSB,
            'DLB': self.DLB,
            'LCB': self.LCB,
            'LSD': self.LSD,
            'ZL': self.ZL,
            'Z0L': self.Z0L,
            'Z0': self.Z0,
            'Z0U': self.Z0U,
            'ZU': self.ZU,
            'USD': self.USD,
            'UCB': self.UCB,
            'DUB': self.DUB,
            'USB': self.USB,
            'UB': self.UB,
            'tolerance_interval': self.tolerance_interval,
            'typical_data_interval': self.typical_data_interval,
            'ordering_valid': self.ordering_valid,
            'search_points': len(self.search_results['datum'])
        })
        self.logger.info(f"""Results updated. 
        Tolerance interval: [{self.Z0L:.4f}, {self.Z0U:.4f}], 
        Typical data interval: [{self.ZL:.4f}, {self.ZU:.4f}] 
        Ordering valid: {self.ordering_valid}""")

    def results(self) -> Dict:
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
        self.logger.info("Retrieving results dictionary...")
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

    def plot_intervals(self, figsize=(12, 8)):
        """
        Plot the Z0 variation and estimated intervals.

        This method visualizes how the central parameter Z0 changes as the data is extended,
        and marks the estimated typical data interval and tolerance interval on the plot.
        It also indicates whether the ordering constraint is satisfied.

        Parameters
        ----------
        figsize : tuple, optional (default=(12, 8))
            Size of the matplotlib figure.

        Notes
        -----
        - The plot shows the Z0 trajectory, interval boundaries, and highlights the ordering constraint status.
        - Useful for diagnostics and for understanding the robustness of the interval estimation.
        """
        self.logger.info("Plotting Z0 variation and intervals...")

        import matplotlib.pyplot as plt
        datums = np.array(self.search_results_clean['datum'])
        z0s = np.array(self.search_results_clean['z0'])
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        sort_idx = np.argsort(datums)
        ax.scatter(datums[sort_idx], z0s[sort_idx], color='k', alpha=0.5, linewidth=1, label='Z0 Variation')
        ax.plot(datums[sort_idx], z0s[sort_idx], color='k', alpha=0.5, linewidth=1)
        ax.scatter([self.ZL], [self.Z0L], marker='v', s=120, color='purple', edgecolor='black', zorder=10, label=f'ZL,Z0L ({self.ZL:.4f},{self.Z0L:.4f})')
        ax.scatter([self.Z0], [self.Z0], marker='s', s=120, color='green', edgecolor='black', zorder=10, label=f'Z0 ({self.Z0:.4f})')
        ax.scatter([self.ZU], [self.Z0U], marker='^', s=120, color='orange', edgecolor='black', zorder=10, label=f'Z0U,ZU ({self.Z0U:.4f},{self.ZU:.4f})')
        ax.axvline(x=self.ZL, color='purple', linestyle='--', alpha=1, linewidth=1)
        ax.axvline(x=self.Z0, color='green', linestyle='-', alpha=1, linewidth=2)
        ax.axvline(x=self.ZU, color='orange', linestyle='--', alpha=1, linewidth=1)
        ax.axhline(y=self.Z0L, color='purple', linestyle=':', alpha=1, linewidth=1)
        ax.axhline(y=self.Z0U, color='orange', linestyle=':', alpha=1, linewidth=1)
        ordering_status = "✓ VALID" if self.ordering_valid else "✗ INVALID"
        tol_interval_str = f"Tolerance Interval: [{self.Z0L:.4f}, {self.Z0U:.4f}]"
        typ_interval_str = f"Typical Data Interval: [{self.ZL:.4f}, {self.ZU:.4f}]"
        ordering_str = f"Ordering Constraint: {ordering_status}"
        ax.plot([], [], ' ', label=tol_interval_str)
        ax.plot([], [], ' ', label=typ_interval_str)
        ax.plot([], [], ' ', label=ordering_str)
        pad = (self.Z0U - self.Z0L) * 0.1
        z0_min, z0_max = self.Z0L - pad, self.Z0U + pad
        ax.set_ylim(z0_min, z0_max)
        ax.set_xlabel('Datum Value', fontsize=12, fontweight='bold')
        ax.set_ylabel('Z0 Value', fontsize=12, fontweight='bold')
        title = 'Z0-Based Interval Estimation'
        if not self.ordering_valid:
            title += ' - ⚠ Ordering Constraint Violated'
        ax.set_title(title, fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        #log summary
        self.logger.info(f"\nZ0 Variation Plot Summary:")
        self.logger.info(f"  Typical data interval: [{self.ZL:.6f}, {self.ZU:.6f}] (width: {self.ZU - self.ZL:.6f})")
        self.logger.info(f"  Tolerance interval: [{self.Z0L:.6f}, {self.Z0U:.6f}] (width: {self.Z0U - self.Z0L:.6f})")
        self.logger.info(f"  Ordering constraint: {'✓ SATISFIED' if self.ordering_valid else '✗ VIOLATED'}")

    def plot(self, figsize=(12, 8)):
        """
        Plot the GDF, PDF, and estimated intervals on the data domain.

        This method visualizes the fitted GDF curve, the probability density function (if available),
        and overlays the estimated typical data interval and tolerance interval.
        It also marks the original data points and key interval boundaries.

        Parameters
        ----------
        figsize : tuple, optional (default=(12, 8))
            Size of the matplotlib figure.

        Notes
        -----
        - The plot provides a comprehensive view of the data, the fitted distribution, and the intervals.
        - Useful for reporting and for visually assessing the coverage and validity of the intervals.
        """
        self.logger.info("Plotting GDF, PDF, and intervals...")

        import matplotlib.pyplot as plt
        x_points = np.array(self.data)
        x_min, x_max = np.min(x_points), np.max(x_points)
        x_pad = (x_max - x_min) * 0.05
        x_min -= x_pad
        x_max += x_pad
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = ax1.twinx()
        
        # gdf points
        gdf_points = f"{self.gdf_name.lower()}_points"
        # ELDF curve (if available)
        gdf_vals = getattr(self.gdf, gdf_points, None)
        smooth_x = getattr(self.gdf, 'di_points_n', None)
        if gdf_vals is not None and smooth_x is not None:
            ax1.plot(smooth_x, gdf_vals, '-', color='blue', linewidth=2.5, alpha=0.9, label=self.gdf_name)
        else:
            ax1.plot(x_points, [self.Z0]*len(x_points), 'o', color='blue', label=self.gdf_name, markersize=4, alpha=0.7)
        # PDF curve (if available)
        pdf_vals = getattr(self.gdf, 'pdf_points', None)
        if pdf_vals is not None and smooth_x is not None:
            ax2.plot(smooth_x, pdf_vals, '-', color='red', linewidth=2.5, alpha=0.9, label='PDF')
            max_pdf = np.max(pdf_vals)
        elif pdf_vals is not None:
            ax2.plot(x_points, pdf_vals, 'o', color='red', label='PDF', markersize=4, alpha=0.7)
            max_pdf = np.max(pdf_vals)
        else:
            max_pdf = 1.0
        
        # Typical Data Interval (ZL to ZU)
        ax1.axvspan(self.ZL, self.ZU, alpha=0.2, color='lightblue', label=f'Typical Data Interval \n[ZL: {self.ZL:.3f}, ZU: {self.ZU:.3f}]')
        # Tolerance Interval (Z0L to Z0U)
        ax1.axvspan(self.Z0L, self.Z0U, alpha=0.20, color='lightgreen', label=f'Tolerance Interval \n[Z0L: {self.Z0L:.3f}, Z0U: {self.Z0U:.3f}]')

        # Critical vertical lines
        ax1.axvline(x=self.ZL, color='orange', linestyle='-.', linewidth=2, alpha=0.8, label=f'ZL={self.ZL:.3f}')
        ax1.axvline(x=self.Z0, color='magenta', linestyle='-.', linewidth=1, alpha=0.9, label=f'Z0={self.Z0:.3f}')
        ax1.axvline(x=self.ZU, color='orange', linestyle='--', linewidth=2, alpha=0.8, label=f'ZU={self.ZU:.3f}')
        ax1.axvline(x=self.Z0L, color='grey', linestyle='-', linewidth=1.5, alpha=0.7, zorder=0)
        ax1.axvline(x=self.Z0U, color='grey', linestyle='-', linewidth=1.5, alpha=0.7, zorder=0)
        # Data bounds
        if self.LB is not None:
            ax1.axvline(x=self.gdf.LB, color='purple', linestyle='-.', linewidth=1, alpha=1, label=f'LB={self.gdf.LB:.3f}')
        if self.UB is not None:
            ax1.axvline(x=self.gdf.UB, color='purple', linestyle='--', linewidth=1, alpha=1, label=f'UB={self.gdf.UB:.3f}')
        # DLB and DUB bounds
        if self.DLB is not None:
            ax1.axvline(x=self.gdf.DLB, color='brown', linestyle='-.', linewidth=1.5, alpha=1, label=f'DLB={self.gdf.LB:.3f}')
        if self.DUB is not None:
            ax1.axvline(x=self.gdf.DUB, color='brown', linestyle='--', linewidth=1.5, alpha=1, label=f'DUB={self.gdf.DUB:.3f}')
        # LSB and USB bounds
        if self.LSB is not None:
            ax1.axvline(x=self.gdf.LSB, color='red', linestyle='-.', linewidth=1, alpha=1, label=f'LSB={self.gdf.LSB:.3f}')
        if self.USB is not None:
            ax1.axvline(x=self.gdf.USB, color='red', linestyle='--', linewidth=1, alpha=1, label=f'USB={self.gdf.USB:.3f}')
        # LCB and UCB bounds
        if self.LCB is not None:
            ax1.axvline(x=self.gdf.LCB, color='blue', linestyle='-', linewidth=1, alpha=1, label=f'LCB={self.gdf.LCB:.3f}')
        if self.UCB is not None:
            ax1.axvline(x=self.gdf.UCB, color='blue', linestyle='--', linewidth=1, alpha=1, label=f'UCB={self.gdf.UCB:.3f}')
        # LSD and USD bounds
        if self.LSD is not None:
            ax1.axvline(x=self.LSD, color='cyan', linestyle='-.', linewidth=1, alpha=1, label=f'LSD={self.LSD:.3f}')
        if self.USD is not None:
            ax1.axvline(x=self.USD, color='cyan', linestyle='--', linewidth=1, alpha=1, label=f'USD={self.USD:.3f}')
        # Rug plot for original data
        data_y_pos = -0.05
        ax1.scatter(x_points, [data_y_pos] * len(x_points), alpha=0.6, s=15, color='black', marker='|')
        ax1.set_xlabel('Data Values', fontsize=12, fontweight='bold')
        ax1.set_ylabel(f'{self.gdf_name} Value', fontsize=12, fontweight='bold', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_ylim(-0.1, 1.05)
        ax1.set_xlim(x_min, x_max)
        ax2.set_ylabel('PDF Value', fontsize=12, fontweight='bold', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(0, max_pdf * 1.1)
        ax2.set_xlim(x_min, x_max)
        ax1.grid(True, alpha=0.3)
        title_text = f'{self.gdf_name} Interval Analysis (Z0 = {self.Z0:.3f})'
        ax1.set_title(title_text, fontsize=12)
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10, borderaxespad=0)
        plt.tight_layout()
        plt.show()
        
        self.logger.info(f"\n{self.gdf_name} Interval Analysis Plot Summary:")
        self.logger.info(f"  Z0 (Gnostic Mode): {self.Z0:.4f}")
        self.logger.info(f"  Tolerance interval: [{self.Z0L:.4f}, {self.Z0U:.4f}] (width: {self.Z0U - self.Z0L:.4f})")
        self.logger.info(f"  Typical data interval: [{self.ZL:.4f}, {self.ZU:.4f}] (width: {self.ZU - self.ZL:.4f})")
        data_in_tolerance = np.sum((x_points >= self.Z0L) & (x_points <= self.Z0U))
        self.logger.info(f"  Data coverage - Tolerance: {data_in_tolerance}/{len(x_points)} ({data_in_tolerance/len(x_points):.1%})")
        data_in_typical = np.sum((x_points >= self.ZL) & (x_points <= self.ZU))
        self.logger.info(f"  Data coverage - Typical: {data_in_typical}/{len(x_points)} ({data_in_typical/len(x_points):.1%})")
        self.logger.info(f"  Total data points: {len(x_points)}")
        self.logger.info(f"  Data range: [{np.min(x_points):.4f}, {np.max(x_points):.4f}]")

    def _flush_memory(self):
        if self.flush:
            self.search_results = {'datum': [], 'z0': [], 'success': []}
        self.logger.info("Flushed data to free memory.")

    def __repr__(self):
        return (f"DataIntervals(gdf={self.gdf_name}, n_points={self.n_points}, "
                f"dense_zone_fraction={self.dense_zone_fraction}, "
                f"dense_points_fraction={self.dense_points_fraction}, "
                f"convergence_window={self.convergence_window}, "
                f"convergence_threshold={self.convergence_threshold}, "
                f"min_search_points={self.min_search_points}, "
                f"boundary_margin_factor={self.boundary_margin_factor}, "
                f"extrema_search_tolerance={self.extrema_search_tolerance}, "
                f"verbose={self.verbose}, flush={self.flush})")