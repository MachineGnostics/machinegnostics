import numpy as np
import warnings
from typing import Optional, Union, Dict
from machinegnostics.magcal import ELDF, EGDF, QLDF, QGDF

class DataIntervals:
    """
    Fast, robust interval estimation for GDF classes.
    """

    def __init__(self, 
                 gdf: Union[ELDF, EGDF, QLDF, QGDF], 
                 verbose: bool = False, 
                 catch: bool = True, 
                 tolerance: float = 1e-5, 
                 early_stopping_steps: int = 25,
                 n_points: int = 500,
                 flush: bool = False):
        self.gdf = gdf
        self.verbose = verbose
        self.catch = catch
        self.tolerance = tolerance
        self.early_stopping_steps = early_stopping_steps
        self.n_points = n_points
        self.flush = flush
        self.params: Dict = {}
        self.z0_dict = {'datum': [], 'Z0': []}

    def _extract_gdf_data(self):
        self.data = self.gdf.data
        self.Z0 = self.gdf.z0
        self.LB = self.gdf.LB
        self.UB = self.gdf.UB
        self.DLB = self.gdf.DLB
        self.DUB = self.gdf.DUB
        # try to get LSB, USB, LCB, UCB if available
        self.LSB = getattr(self.gdf, 'LSB', None)
        self.USB = getattr(self.gdf, 'USB', None)
        self.LCB = getattr(self.gdf, 'LCB', None)
        self.UCB = getattr(self.gdf, 'UCB', None)

        if self.catch:
            self.params = dict(self.gdf.params)
        if self.verbose:
            print(f'DataIntervals: Data length: {len(self.data)}, Z0: {self.Z0}')

    def _extend_and_record(self, datum: float):
        extended_data = np.append(self.data, datum)
        gdf_type = type(self.gdf)
        # Only pass valid constructor arguments
        kwargs = {
            'verbose': self.verbose,
            'wedf': getattr(self.gdf, 'wedf', False),
            'S': getattr(self.gdf, 'S', None),
            'varS': getattr(self.gdf, 'varS', None),
            'flush': getattr(self.gdf, 'flush', True),
            'catch': getattr(self.gdf, 'catch', True),
            'z0_optimize': getattr(self.gdf, 'z0_optimize', False),
            'tolerance': getattr(self.gdf, 'tolerance', 1e-5),
            'data_form': getattr(self.gdf, 'data_form', None),
            'n_points': getattr(self.gdf, 'n_points', 20),
            'opt_method': getattr(self.gdf, 'opt_method', None),
            'max_data_size': getattr(self.gdf, 'max_data_size', None),
        }
        # Remove None values (not all GDFs use all args)
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        gdf_extended = gdf_type(**kwargs)
        gdf_extended.fit(data=extended_data, plot=False)
        self.z0_dict['datum'].append(datum)
        self.z0_dict['Z0'].append(gdf_extended.z0)

    def _scan_intervals(self):
        # Vectorized scan over lower and upper ranges
        lower_range = np.linspace(self.Z0, self.LB, self.n_points)
        upper_range = np.linspace(self.Z0, self.UB, self.n_points)
        for datum in np.concatenate([lower_range, upper_range]):
            self._extend_and_record(datum)
            if self._early_stopping_check():
                break

    def _early_stopping_check(self):
        z0s = np.array(self.z0_dict['Z0'])
        if len(z0s) < self.early_stopping_steps + 1:
            return False
        changes = np.abs(np.diff(z0s)[-self.early_stopping_steps:])
        if np.all(changes < self.tolerance):
            if self.verbose:
                print('DataIntervals: Early stopping triggered.')
            return True
        return False

    def _extract_interval_params(self):
        z0s = np.array(self.z0_dict['Z0'])
        datums = np.array(self.z0_dict['datum'])
        min_idx, max_idx = np.argmin(z0s), np.argmax(z0s)
        self.Z0L, self.ZL = z0s[min_idx], datums[min_idx]
        self.Z0U, self.ZU = z0s[max_idx], datums[max_idx]
        if self.catch:
            self.params.update({'ZL': self.ZL, 'Z0L': self.Z0L, 'Z0': self.Z0, 'Z0U': self.Z0U, 'ZU': self.ZU})
        if self.verbose:
            print(f'DataIntervals: Intervals: Z0L={self.Z0L}, Z0U={self.Z0U}, ZL={self.ZL}, ZU={self.ZU}')

    def _interval_validation(self):
        valid = self.ZL < self.Z0L < self.Z0 < self.Z0U < self.ZU
        if not valid:
            # if self.verbose:
            #     print('DataIntervals: Interval validation failed!')
            self.params['interval_validation'] = False
        else:
            if self.verbose:
                print('DataIntervals: Interval validation succeeded.')
            self.params['interval_validation'] = True
        return valid

    def _interval_correction(self):
        z0s = np.array(self.z0_dict['Z0'])
        datums = np.array(self.z0_dict['datum'])
        sorted_idx = np.argsort(z0s)
        for i in range(len(z0s)):
            for j in range(len(z0s)-1, i, -1):
                self.Z0L, self.ZL = z0s[sorted_idx[i]], datums[sorted_idx[i]]
                self.Z0U, self.ZU = z0s[sorted_idx[j]], datums[sorted_idx[j]]
                self.params.update({'ZL': self.ZL, 'Z0L': self.Z0L, 'Z0': self.Z0, 'Z0U': self.Z0U, 'ZU': self.ZU})
                if self._interval_validation():
                    if self.verbose:
                        print(f'DataIntervals: Correction succeeded: min={i}, max={j}')
                    return
        warnings.warn('DataIntervals: Correction failed for all candidates.')

    def fit(self, plot: bool = False):
        self._extract_gdf_data()
        self._scan_intervals()
        self._extract_interval_params()
        if not self._interval_validation():
            self._interval_correction()
        # final interval validation
        if not self._interval_validation():
            warnings.warn('DataIntervals: Final interval validation failed after correction. Check data or initialization parameters.')
        if plot:
            self.plot()
            self.plot_intervals()

        if self.flush:
            self._flush_memory()

    def results(self) -> Dict:
        return dict(self.params)

    def plot_intervals(self, figsize=(12, 8)):
        """
        Plot Z0 variation and interval boundaries (IntveEngine style).
        """
        if self.flush and not hasattr(self, 'z0_dict'):
            raise ValueError("DataIntervals: No data available for plotting. Set flush=False during initialization.")
        
        import matplotlib.pyplot as plt

        datum_vals = np.array(self.z0_dict['datum'])
        z0_vals = np.array(self.z0_dict['Z0'])

        if len(datum_vals) == 0 or len(z0_vals) == 0:
            print("No valid data for plotting")
            return

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Main Z0 variation curve
        sort_idx = np.argsort(datum_vals)
        ax.scatter(datum_vals[sort_idx], z0_vals[sort_idx], color='k', alpha=0.5, linewidth=1, label='Z0 Variation')

        # Critical points
        ax.scatter([self.ZL], [self.Z0L], marker='v', s=120, color='purple', edgecolor='black', zorder=10, label=f'ZL,Z0L ({self.ZL:.4f},{self.Z0L:.4f})')
        ax.scatter([self.Z0], [self.Z0], marker='s', s=120, color='green', edgecolor='black', zorder=10, label=f'Z0 ({self.Z0:.4f})')
        ax.scatter([self.ZU], [self.Z0U], marker='^', s=120, color='orange', edgecolor='black', zorder=10, label=f'Z0U,ZU ({self.Z0U:.4f},{self.ZU:.4f})')

        # Reference lines
        ax.axvline(x=self.ZL, color='purple', linestyle='--', alpha=0.7, linewidth=1)
        ax.axvline(x=self.Z0, color='green', linestyle='-', alpha=0.8, linewidth=2)
        ax.axvline(x=self.ZU, color='orange', linestyle='--', alpha=0.7, linewidth=1)
        ax.axhline(y=self.Z0L, color='purple', linestyle=':', alpha=0.7, linewidth=1)
        ax.axhline(y=self.Z0U, color='orange', linestyle=':', alpha=0.7, linewidth=1)

        # Interval info and ordering status
        ordering_valid = (self.ZL < self.Z0L < self.Z0 < self.Z0U < self.ZU)
        ordering_status = "✓ VALID" if ordering_valid else "✗ INVALID"
        tol_interval_str = f"Tolerance Interval: [{self.Z0L:.4f}, {self.Z0U:.4f}]"
        typ_interval_str = f"Typical Data Interval: [{self.ZL:.4f}, {self.ZU:.4f}]"
        ordering_str = f"Ordering Constraint: {ordering_status}"

        ax.plot([], [], ' ', label=tol_interval_str)
        ax.plot([], [], ' ', label=typ_interval_str)
        ax.plot([], [], ' ', label=ordering_str)

        # y-axis limits
        z0_min, z0_max = self.Z0L - 0.1 * abs(self.Z0L), self.Z0U + 0.1 * abs(self.Z0U)
        ax.set_ylim(z0_min, z0_max)

        ax.set_xlabel('Datum Value', fontsize=12, fontweight='bold')
        ax.set_ylabel('Z0 Value', fontsize=12, fontweight='bold')
        title = 'Z0-Based Interval Estimation'
        if not ordering_valid:
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
            print(f"  Ordering constraint: {'✓ SATISFIED' if ordering_valid else '✗ VIOLATED'}")

    def plot(self, figsize=(12, 8)):
        """
        Comprehensive interval analysis plot with tolerance and typical data intervals,
        ELDF curve, PDF (if available), filled zones, critical points, and rug plot.
        """
        if self.flush and not hasattr(self, 'z0_dict'):
            raise ValueError("DataIntervals: No data available for plotting. Set flush=False during initialization.")   
        
        import matplotlib.pyplot as plt
        import numpy as np

        # Use fitted GDF data
        x_points = np.array(self.data)
        eldf_vals = getattr(self.gdf, 'eldf_points', None)
        pdf_vals = getattr(self.gdf, 'pdf_points', None)
        wedf_vals = getattr(self.gdf, 'wedf_points', None)

        # Smooth curve data if available
        smooth_x = getattr(self.gdf, 'di_points_n', None)
        smooth_eldf = getattr(self.gdf, 'eldf_points', None)
        smooth_pdf = getattr(self.gdf, 'pdf_points', None)

        # X-axis range with padding
        x_min, x_max = np.min(x_points), np.max(x_points)
        x_pad = (x_max - x_min) * 0.05
        x_min -= x_pad
        x_max += x_pad
        x_fine = np.linspace(x_min, x_max, 1000)

        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = ax1.twinx()

        # Plot ELDF curve
        if eldf_vals is not None and smooth_x is not None:
            ax1.plot(smooth_x, smooth_eldf, '-', color='blue', linewidth=2.5, alpha=0.9, label='ELDF')
        else:
            ax1.plot(x_points, [self.Z0]*len(x_points), 'o', color='blue', label='ELDF', markersize=4, alpha=0.7)

        # Plot PDF curve
        if pdf_vals is not None and smooth_x is not None:
            ax2.plot(smooth_x, smooth_pdf, '-', color='red', linewidth=2.5, alpha=0.9, label='PDF')
            max_pdf = np.max(smooth_pdf)
        elif pdf_vals is not None:
            ax2.plot(x_points, pdf_vals, 'o', color='red', label='PDF', markersize=4, alpha=0.7)
            max_pdf = np.max(pdf_vals)
        else:
            max_pdf = 1.0

        # Tolerance Interval (Z0L to Z0U) - Light Green
        ax1.axvspan(self.Z0L, self.Z0U, alpha=0.15, color='lightgreen',
                    label=f'Tolerance Interval [Z0L: {self.Z0L:.3f}, Z0U: {self.Z0U:.3f}]')

        # Typical Data Interval (ZL to ZU) - Light Blue
        ax1.axvspan(self.ZL, self.ZU, alpha=0.15, color='lightblue',
                    label=f'Typical Data Interval [ZL: {self.ZL:.3f}, ZU: {self.ZU:.3f}]')

        # Critical vertical lines (ZL, Z0, ZU) with legend
        ax1.axvline(x=self.ZL, color='purple', linestyle='--', linewidth=2, alpha=0.8, label=f'ZL={self.ZL:.3f}')
        ax1.axvline(x=self.Z0, color='magenta', linestyle='-.', linewidth=1, alpha=0.9, label=f'Z0={self.Z0:.3f}')
        ax1.axvline(x=self.ZU, color='orange', linestyle='--', linewidth=2, alpha=0.8, label=f'ZU={self.ZU:.3f}')

        # Vertical grey lines for Z0L, Z0U (no legend)
        ax1.axvline(x=self.Z0L, color='grey', linestyle='-', linewidth=1.5, alpha=0.7, zorder=0)
        ax1.axvline(x=self.Z0U, color='grey', linestyle='-', linewidth=1.5, alpha=0.7, zorder=0)

        # Vertical grey lines for LSB, USB, CLB, CUB if available (no legend)
        for bound_name in ['LSB', 'USB', 'CLB', 'CUB']:
            bound_val = getattr(self.gdf, bound_name, None)
            if bound_val is not None:
                ax1.axvline(x=bound_val, color='grey', linestyle=':', linewidth=1.2, alpha=0.7, zorder=0)

        # Data bounds if available (with legend)
        if hasattr(self.gdf, 'LB') and self.gdf.LB is not None:
            ax1.axvline(x=self.gdf.LB, color='purple', linestyle='--', linewidth=1, alpha=0.6, label=f'LB={self.gdf.LB:.3f}')
        if hasattr(self.gdf, 'UB') and self.gdf.UB is not None:
            ax1.axvline(x=self.gdf.UB, color='brown', linestyle='--', linewidth=1, alpha=0.6, label=f'UB={self.gdf.UB:.3f}')

        # Rug plot for original data
        data_y_pos = -0.05
        ax1.scatter(x_points, [data_y_pos] * len(x_points), alpha=0.6, s=15, color='black', marker='|')

        # Axis labels and limits
        ax1.set_xlabel('Data Values', fontsize=12, fontweight='bold')
        ax1.set_ylabel('ELDF Value', fontsize=12, fontweight='bold', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_ylim(-0.1, 1.05)
        ax1.set_xlim(x_min, x_max)

        ax2.set_ylabel('PDF Value', fontsize=12, fontweight='bold', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(0, max_pdf * 1.1)
        ax2.set_xlim(x_min, x_max)

        ax1.grid(True, alpha=0.3)

        # Title
        title_text = f'ELDF Interval Analysis (Z0 = {self.Z0:.3f})'
        ax1.set_title(title_text, fontsize=12)

        # Legend (exclude grey lines)
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles, labels, loc='upper left', fontsize=10, bbox_to_anchor=(0.02, 0.98))

        plt.tight_layout()
        plt.show()

        # Print summary
        if self.verbose:
            print(f"\nELDF Interval Analysis Plot Summary:")
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
        # delete large attributes to free memory
        if self.flush:
            del self.z0_dict
        
        if self.verbose:
            print("DataIntervals: Flushed data to free memory.")
