'''
DataIntervals

Interval Analysis Engine

Author: Nirmal Parmar
Machine Gnostics
'''

import numpy as np
from typing import Optional, Union, Dict
from scipy.signal import savgol_filter, find_peaks
from machinegnostics.magcal import ELDF, EGDF, QLDF, QGDF, DataCluster

class DataIntervals:
    """
    Robust interval estimation for GDF classes with adaptive search, diagnostics, and ordering constraint.
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
        if self.verbose:
            print(f"DataIntervals: Warning: {message}")
        if self.catch:
            self.params['warnings'].append(message)
    
    def _add_error(self, message: str):
        self.params['errors'].append(message)
        if self.verbose:
            print(f"DataIntervals: Error: {message}")
        if self.catch:
            self.params['errors'].append(message)

    def _extract_gdf_data(self):
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

            if self.verbose:
                print(f"DataIntervals: Initialized with {self.params['gdf_type']} | Data size: {self.params['data_size']} | Z0: {self.Z0:.6f}")

        except Exception as e:
            self._add_error(f"DataIntervals: Failed to extract GDF data: {e}")
            return
        
    def _argument_validation(self):
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
        if self.verbose:
            print("DataIntervals: Initial parameters stored.")

    def _reset_results(self):
        self.ZL = None
        self.Z0L = None
        self.ZU = None
        self.Z0U = None
        self.tolerance_interval = None
        self.typical_data_interval = None
        self.ordering_valid = None

    def fit(self, plot: bool = False):
        import time
        start_time = time.time()
        try:
            self._argument_validation()

            if self.verbose:
                print("\nDataIntervals: Fit process started.")
            self._reset_results()
    
            # Scan intervals and extract boundaries
            self._scan_intervals()
            self._extract_intervals_with_ordering()
    
            # Check ordering constraint
            if not self.ordering_valid:
                msg = ("Interval ordering constraint violated. "
                       "Try setting 'wedf=False', or setting 'gnostic_filter=True', or increasing 'n_points', or adjusting thresholds for sensitivity.")
                self._add_warning(msg)
    
            # Update parameters and optionally plot
            self._update_params()
            if plot:
                self.plot()
                self.plot_intervals()
    
            # Optionally flush memory
            if self.flush:
                self._flush_memory()
    
            if self.verbose:
                elapsed = time.time() - start_time
                print(f"DataIntervals: Fit process completed in {elapsed:.2f} seconds.")
                print(f"DataIntervals: Ordering valid: {self.ordering_valid}")
                print(f"DataIntervals: Tolerance interval: [{self.Z0L:.4f}, {self.Z0U:.4f}]")
                print(f"DataIntervals: Typical data interval: [{self.ZL:.4f}, {self.ZU:.4f}]")
        except Exception as e:
            err_msg = f"DataIntervals: Fit failed: {e}"
            if self.verbose:
                print(f"DataIntervals: ERROR: {err_msg}")
            self._add_error(err_msg)
            raise

    def _scan_intervals(self):
        try:
            if self.verbose:
                print("DataIntervals: Scanning intervals...")
            # Adaptive search: dense near Z0, sparse near LB/UB
            lower_points = self._generate_search_points('lower')
            upper_points = self._generate_search_points('upper')
            for datum in np.concatenate([lower_points, upper_points]):
                z0_val = self._compute_z0_with_extended_datum(datum)
                self.search_results['datum'].append(datum)
                self.search_results['z0'].append(z0_val)
                self.search_results['success'].append(True)
                if self.verbose:
                    # print on 50th point
                    if len(self.search_results['datum']) % 50 == 0:
                        print(f"  Datum: {datum:.4f} | New Z0: {z0_val:.6f}")
                if self._check_convergence():
                    if self.verbose:
                        print(f"DataIntervals: Early stopping at datum={datum:.4f}")
                    break
        except Exception as e:
            self._add_error(f"DataIntervals: Scanning intervals failed: {e}")
            return

    def _generate_search_points(self, direction: str) -> np.ndarray:
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
        z0s = np.array(self.search_results['z0'])
        if len(z0s) < self.convergence_window + self.min_search_points:
            return False
        window = z0s[-self.convergence_window:]
        if np.std(window) < self.convergence_threshold:
            return True
        return False

    def _get_z0s_main_cluster(self, z0s: np.ndarray, datums: np.ndarray) -> np.ndarray:
        # try:
        if self.verbose:
            print("DataIntervals: Extracting main Z0 cluster...")
        
        # 4 less data points - skip clustering
        if len(z0s) <= 4 or len(datums) < 4:
            self._add_warning("Insufficient data points for clustering. Returning all values.")
            return z0s, datums

        # Fit ELDF to z0s for clustering
        eldf_cluster = ELDF(catch=False, wedf=False, verbose=False)
        eldf_cluster.fit(z0s)
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
    
        # except Exception as e:
            # self._add_warning(f"Cluster-based Z0 extraction failed: {e}. Using all Z0 values.")
            # return np.array(self.search_results['z0']), np.array(self.search_results['datum'])

    def _extract_intervals_with_ordering(self):
        datums = np.array(self.search_results['datum'])
        z0s = np.array(self.search_results['z0'])

        if self.gnostic_filter:
            if self.verbose:
                print("DataIntervals: Applying gnostic filtering to Z0 values...")
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
        self.tolerance_interval = self.Z0U - self.Z0L
        self.typical_data_interval = self.ZU - self.ZL

    def _find_valid_extrema_with_ordering(self, datums, z0s):
        # Try combinations to satisfy ordering constraint
        lower_mask = datums < self.Z0
        upper_mask = datums > self.Z0
        lower_datum = datums[lower_mask]
        lower_z0 = z0s[lower_mask]
        upper_datum = datums[upper_mask]
        upper_z0 = z0s[upper_mask]
        n_candidates = min(5, len(lower_datum), len(upper_datum))
        found = False
        for i in range(n_candidates):
            zl, z0l = lower_datum[i], lower_z0[i]
            zu, z0u = upper_datum[-(i+1)], upper_z0[-(i+1)]
            if zl < z0l < self.Z0 < z0u < zu:
                self.ZL, self.Z0L, self.ZU, self.Z0U = zl, z0l, zu, z0u
                self.ordering_valid = True
                found = True
                break
        if not found:
            # Fallback: use initial extrema
            min_idx = np.argmin(z0s)
            max_idx = np.argmax(z0s)
            self.ZL, self.Z0L, self.ZU, self.Z0U = datums[min_idx], z0s[min_idx], datums[max_idx], z0s[max_idx]
            self.ordering_valid = False
        if self.verbose:
            print(f"DataIntervals: Ordering constraint {'satisfied' if self.ordering_valid else 'NOT satisfied'}.")

    def _update_params(self):
        self.params.update({
            'LB': self.LB,
            'LSB': self.LSB,
            'DLB': self.DLB,
            'LCB': self.LCB,
            'ZL': self.ZL,
            'Z0L': self.Z0L,
            'Z0': self.Z0,
            'Z0U': self.Z0U,
            'ZU': self.ZU,
            'UCB': self.UCB,
            'DUB': self.DUB,
            'USB': self.USB,
            'UB': self.UB,
            'tolerance_interval': self.tolerance_interval,
            'typical_data_interval': self.typical_data_interval,
            'ordering_valid': self.ordering_valid,
            'search_points': len(self.search_results['datum'])
        })
        if self.verbose:
            print(f"""DataIntervals: Results updated. 
        Tolerance interval: [{self.Z0L:.4f}, {self.Z0U:.4f}], 
        Typical data interval: [{self.ZL:.4f}, {self.ZU:.4f}] 
        Ordering valid: {self.ordering_valid}""")

    def results(self) -> Dict:
        results = {
            'LB': float(self.LB) if self.LB is not None else None,
            'LSB': float(self.LSB) if self.LSB is not None else None,
            'DLB': float(self.DLB) if self.DLB is not None else None,
            'LCB': float(self.LCB) if self.LCB is not None else None,
            'ZL': float(self.ZL) if self.ZL is not None else None,
            'Z0L': float(self.Z0L) if self.Z0L is not None else None,
            'Z0': float(self.Z0) if self.Z0 is not None else None,
            'Z0U': float(self.Z0U) if self.Z0U is not None else None,
            'ZU': float(self.ZU) if self.ZU is not None else None,
            'UCB': float(self.UCB) if self.UCB is not None else None,
            'DUB': float(self.DUB) if self.DUB is not None else None,
            'USB': float(self.USB) if self.USB is not None else None,
            'UB': float(self.UB) if self.UB is not None else None
        }
        return results

    def plot_intervals(self, figsize=(12, 8)):
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
        if self.verbose:
            print(f"\nZ0 Variation Plot Summary:")
            print(f"  Typical data interval: [{self.ZL:.6f}, {self.ZU:.6f}] (width: {self.ZU - self.ZL:.6f})")
            print(f"  Tolerance interval: [{self.Z0L:.6f}, {self.Z0U:.6f}] (width: {self.Z0U - self.Z0L:.6f})")
            print(f"  Ordering constraint: {'✓ SATISFIED' if self.ordering_valid else '✗ VIOLATED'}")

    def plot(self, figsize=(12, 8)):
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
        ax1.axvline(x=self.ZL, color='purple', linestyle='--', linewidth=2, alpha=0.8, label=f'ZL={self.ZL:.3f}')
        ax1.axvline(x=self.Z0, color='magenta', linestyle='-.', linewidth=1, alpha=0.9, label=f'Z0={self.Z0:.3f}')
        ax1.axvline(x=self.ZU, color='orange', linestyle='--', linewidth=2, alpha=0.8, label=f'ZU={self.ZU:.3f}')
        ax1.axvline(x=self.Z0L, color='grey', linestyle='-', linewidth=1.5, alpha=0.7, zorder=0)
        ax1.axvline(x=self.Z0U, color='grey', linestyle='-', linewidth=1.5, alpha=0.7, zorder=0)
        # Data bounds
        if self.LB is not None:
            ax1.axvline(x=self.gdf.LB, color='purple', linestyle='--', linewidth=1, alpha=1, label=f'LB={self.gdf.LB:.3f}')
        if self.UB is not None:
            ax1.axvline(x=self.gdf.UB, color='brown', linestyle='--', linewidth=1, alpha=1, label=f'UB={self.gdf.UB:.3f}')
        # DLB and DUB bounds
        if self.DLB is not None:
            ax1.axvline(x=self.gdf.DLB, color='purple', linestyle='-', linewidth=1.5, alpha=1, label=f'DLB={self.gdf.LB:.3f}')
        if self.DUB is not None:
            ax1.axvline(x=self.gdf.DUB, color='brown', linestyle='-', linewidth=1.5, alpha=1, label=f'DUB={self.gdf.LB:.3f}')
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
        if self.verbose:
            print(f"\n{self.gdf_name} Interval Analysis Plot Summary:")
            print(f"  Z0 (Gnostic Mode): {self.Z0:.4f}")
            print(f"  Tolerance interval: [{self.Z0L:.4f}, {self.Z0U:.4f}] (width: {self.Z0U - self.Z0L:.4f})")
            print(f"  Typical data interval: [{self.ZL:.4f}, {self.ZU:.4f}] (width: {self.ZU - self.ZL:.4f})")
            data_in_tolerance = np.sum((x_points >= self.Z0L) & (x_points <= self.Z0U))
            print(f"  Data coverage - Tolerance: {data_in_tolerance}/{len(x_points)} ({data_in_tolerance/len(x_points):.1%})")
            data_in_typical = np.sum((x_points >= self.ZL) & (x_points <= self.ZU))
            print(f"  Data coverage - Typical: {data_in_typical}/{len(x_points)} ({data_in_typical/len(x_points):.1%})")
            print(f"  Total data points: {len(x_points)}")
            print(f"  Data range: [{np.min(x_points):.4f}, {np.max(x_points):.4f}]")

    def _flush_memory(self):
        if self.flush:
            self.search_results = {'datum': [], 'z0': [], 'success': []}
        if self.verbose:
            print("DataIntervals: Flushed data to free memory.")
