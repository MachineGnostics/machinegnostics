'''
Marginal Interval Analysis

Take care of end-2-end gnostic process. Primarily work with ELDF.

This module implements the `DataIntervals` class, which provides robust, adaptive, and diagnostic interval estimation for GDF classes such as ELDF, EGDF, QLDF, and QGDF. It estimates meaningful data intervals (such as tolerance and typical intervals) based on the behavior of the GDF's central parameter (Z0) as the data is extended, while enforcing ordering constraints and providing detailed diagnostics.

Author: Nirmal Parmar
Machine Gnostics
'''

import numpy as np
import warnings
from machinegnostics.magcal import ELDF, EGDF, DataHomogeneity, DataIntervals

class IntervalAnalysis:
    def __init__(self,
                DLB: float = None,
                DUB: float = None,
                LB: float = None,
                UB: float = None,
                S: str = 'auto',
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
                flush: bool = True,
                dense_zone_fraction: float = 0.4,
                dense_points_fraction: float = 0.7,
                convergence_window: int = 15,
                convergence_threshold: float = 0.000001,
                min_search_points: int = 30,
                boundary_margin_factor: float = 0.001,
                extrema_search_tolerance: float = 0.000001,
                gdf_recompute: bool = False,
                gnostic_filter: bool = False):
        
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
        self._fitted = False

        self.params = {}
        self.params['error'] = []
        self.params['warnings'] = []

    def _add_warning(self, warning: str):
        self.params['warnings'].append(warning)
        if self.verbose:
            print(f'IntervalAnalysis: Warning: {warning}')
    
    def _add_error(self, error: str):
        self.params['error'].append(error)
        if self.verbose:
            print(f'IntervalAnalysis: Error: {error}')

    def _input_data_check(self, data: np.ndarray):
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a numpy array.")
        if data.ndim != 1:
            raise ValueError("Data must be a 1D array.")
        if data.size < 4:
            raise ValueError("Data must contain at least 4 elements.")
        # no NaN or Inf values
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            raise ValueError("Data contains NaN or Inf values.")
        
    def _check_egdf_homogeneity(self, egdf: EGDF):
        # check homogeneity
        if self.homogeneous:
            self.dh = DataHomogeneity(gdf=egdf, verbose=self.verbose)
            is_homogeneous = self.dh.fit()
            if not is_homogeneous:
                warning_msg = "Data is not homogeneous. Interval estimation may be unreliable."
                self._add_warning(warning_msg)
                if self.catch:
                    self.params['warnings'].append(warning_msg)
                    self.params['DataHomogeneity'] = self.dh.params.copy()
                else:
                    warnings.warn(warning_msg)
        else:
            warning_msg = "Homogeneity check is disabled. Proceeding without checking."
            self._add_warning(warning_msg)
            is_homogeneous = True
            if self.catch:
                self.params['warnings'].append(warning_msg)
            else:
                warnings.warn(warning_msg)
        return is_homogeneous

    def fit(self, data: np.ndarray, plot: bool = False):
        if self.verbose:
            print("IntervalAnalysis: Starting fit process...")

        # check input data
        if self.verbose:
            print("IntervalAnalysis: Checking input data...")
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
        if self.verbose:
            print("IntervalAnalysis: Estimating EGDF...")
        self._egdf = EGDF(**kwargs)
        self._egdf.fit(data)
        if self.catch:
            self.params['EGDF'] = self._egdf.params.copy()

        # check homogeneity
        if self.verbose:
            print("IntervalAnalysis: Checking data homogeneity...")
        is_homogeneous = self._check_egdf_homogeneity(self._egdf)

        # data must be homogeneous
        if not is_homogeneous:
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
        is_homogeneous = self._check_egdf_homogeneity(self._egdf)

        # final check on homogeneity, raise warning, that cannot converted to homogeneous, check data
        if not is_homogeneous:
            warning_msg = "Data is not homogeneous after re-estimation."
            self._add_warning(warning_msg)
            if self.catch:
                self.params['warnings'].append(warning_msg)
                self.params['DataHomogeneity'] = self.dh.params.copy()
            else:
                warnings.warn(warning_msg)

        # estimate ELDF
        kwargs_el = {
            'DLB': self.DLB,
            'DUB': self.DUB,
            'LB': self.LB,
            'UB': self.UB,
            'S': self.S,
            'z0_optimize': self.z0_optimize,
            'tolerance': self.tolerance,
            'data_form': self.data_form,
            'n_points': self.n_points,
            'homogeneous': True, # ELDF always assumes homogeneous data
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

        # estimate intervals with DataIntervals, minimum compute settings
        if self.verbose:
            print("IntervalAnalysis: Estimating data intervals...")
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
        self._di = DataIntervals(**di_kwargs)
        self._di.fit()

        if self.catch:
            self.params['DataIntervals'] = self._di.params.copy()
        
        # fit status
        self._fitted = True
        if self.catch:
            self.params['fitted'] = self._fitted

        # if plot is True, generate diagnostic plots
        if plot:
            if self.verbose:
                print("IntervalAnalysis: Generating diagnostic plots...")
            self._di.plot()

        if self.verbose:
            print("IntervalAnalysis: Fit process completed.")

    def results(self):
        data_certification = self._di.results()
        return data_certification

    def plot(self, GDF: bool = True, intervals: bool = True):
        if hasattr(self, '_di'):
            if GDF:
                self._eldf.plot()
            if intervals:
                self._di.plot_intervals()
                self._di.plot()