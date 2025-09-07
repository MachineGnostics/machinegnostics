'''
Base Interval Analysis for EGDF

**Interval Analysis - Critical Points:**
    
    5. **Z0**: Central point where PDF reaches global maximum and EGDF ≈ 0.5. This represents the
       distribution's central tendency and inflection point.
    
    6. **Z0L and Z0U**: Lower and upper bounds of the tolerance interval around Z0.
    
    7. **ZL and ZU**: Bounds defining the interval of typical data values.
'''

import numpy as np
import warnings
from machinegnostics.magcal.gdf.egdf import EGDF
from machinegnostics.magcal.gdf.base_eg_ma import BaseMarginalAnalysisEGDF
from machinegnostics.magcal.gdf.homogeneity import DataHomogeneity
from machinegnostics.magcal.gdf.intv_engine import IntveEngine

class BaseIntervalAnalysisEGDF(BaseMarginalAnalysisEGDF):
    """
    Base class for interval analysis in EGDF.
    
    Attributes:
        DLB (float): Lower bound of the data range.
        DUB (float): Upper bound of the data range.
        LSB (float): Lower bound of the sample.
        USB (float): Upper bound of the sample.
        z0 (float): Central point where PDF reaches global maximum and EGDF ≈ 0.5.
    """
    
    def __init__(self,
                data: np.ndarray,
                DLB: float = None,
                DUB: float = None,
                LB: float = None,
                UB: float = None,
                S = 'auto',
                z0_optimize: bool = True, # NOTE EGDF specific
                tolerance: float = 1e-9, # NOTE for intv specific
                data_form: str = 'a',
                n_points: int = 1000, # NOTE for intv specific
                homogeneous: bool = True,
                catch: bool = True,
                weights: np.ndarray = None,
                wedf: bool = True,
                opt_method: str = 'L-BFGS-B',
                verbose: bool = False,
                max_data_size: int = 1000,
                flush: bool = True, n_points_per_direction: int = 1000, # intv engine specific
                estimate_sample_bounds: bool = False,
                estimate_cluster_bounds: bool = False,
                sample_bound_tolerance: float = 0.1,
                max_iterations: int = 1000, # NOTE for intv specific
                early_stopping_steps: int = 10,
                estimating_rate: float = 0.01, # NOTE for intv specific
                cluster_threshold: float = 0.05,
                get_clusters: bool = False, # NOTE for intv specific
                dense_zone_fraction: float = 0.4,
                dense_points_fraction: float = 0.7,
                convergence_window: int = 15,
                convergence_threshold: float = 1e-3, # EGDF INTV specific
                min_search_points: int = 30,
                boundary_margin_factor: float = 0.01, # EGDF INTV specific
                extrema_search_tolerance: float = 1e-3,): # EGDF INTV specifi
        super().__init__(data=data, 
                         sample_bound_tolerance=sample_bound_tolerance, 
                         max_iterations=max_iterations, 
                         early_stopping_steps=early_stopping_steps,
                         estimating_rate=estimating_rate, 
                         cluster_threshold=cluster_threshold, 
                         get_clusters=get_clusters, 
                         DLB=DLB, 
                         DUB=DUB, 
                         LB=LB, 
                         UB=UB,
                         S=S, 
                         z0_optimize=z0_optimize,
                         tolerance=tolerance, 
                         data_form=data_form, 
                         n_points=n_points, 
                         homogeneous=homogeneous, 
                         catch=catch, 
                         weights=weights,
                         wedf=wedf, 
                         opt_method=opt_method, 
                         verbose=verbose, 
                         max_data_size=max_data_size, 
                         flush=flush)
        # Initialize interval-specific attributes
        self.Z0 = None
        self.Z0L = None
        self.Z0U = None
        self.ZL = None
        self.ZU = None
        self.estimate_sample_bounds = estimate_sample_bounds
        self.estimate_cluster_bounds = estimate_cluster_bounds
        self.params = {}

        # interval engine parameters
        self.n_points_per_direction = n_points_per_direction
        self.dense_zone_fraction = dense_zone_fraction
        self.dense_points_fraction = dense_points_fraction
        self.convergence_window = convergence_window
        self.convergence_threshold = convergence_threshold
        self.min_search_points = min_search_points
        self.boundary_margin_factor = boundary_margin_factor
        self.extrema_search_tolerance = extrema_search_tolerance

        # fit status
        self._fitted = False

        # input validation
        if not isinstance(self.estimate_sample_bounds, bool):
            raise ValueError("estimate_sample_bounds must be a boolean.")
        if not isinstance(self.estimate_cluster_bounds, bool):
            raise ValueError("estimate_cluster_bounds must be a boolean.")
        # if get_clusters is True, then estimate_cluster_bounds must be True
        if self.get_clusters and not self.estimate_cluster_bounds:
            raise ValueError(
                "Invalid parameter combination: 'get_clusters=True' requires 'estimate_cluster_bounds=True'. "
                "To get cluster information, cluster bounds must first be estimated. "
                "Please set 'estimate_cluster_bounds=True' or set 'get_clusters=False'."
                )

        # max data size validation
        if len(self.data) > self.max_data_size:
            warnings.warn(f"Data length ({len(self.data)}) exceeds max_data_size ({self.max_data_size}). Set max_data_size to a larger value if hardware allows. OR set `flush=True` and `catch=False` to clear data after each iteration.")

    def _create_extended_egdf_intv(self, datum):
        """Create EGDF with extended data including the given datum."""
        data = self.init_egdf.data

        data_extended = np.append(data, datum)

        egdf_extended = EGDF(
            data=data_extended,
            tolerance=self.tolerance,
            data_form=self.data_form,
            n_points=self.n_points,
            homogeneous=self.homogeneous,
            S=self.S_opt,
            z0_optimize=self.z0_optimize,
            LB=self.LB,
            UB=self.UB,
            catch=False,
            weights=self.weights,
            wedf=self.wedf,
            opt_method=self.opt_method,
            verbose=False,
            max_data_size=self.max_data_size,
            flush=False
        )
        
        egdf_extended.fit(plot=False)
        return egdf_extended


    def _get_intv(self, decimals: int=2) -> dict:
        """
        Get the interval values ZL, Z0L, Z0, Z0U, and ZU after fitting.
        
        Returns:
        --------
        dict
            Dictionary containing the interval values.
        """
        if not hasattr(self, 'z0l') or not hasattr(self, 'z0u'):
            raise ValueError("Interval values have not been computed yet. First fit the model for interval analysis.")

        return {
            'LB': float(np.round(self.LB, decimals)),
            'LSB': float(np.round(self.LSB, decimals)) if hasattr(self, 'LSB') else None,
            'DLB': float(np.round(self.DLB, decimals)),
            'CLB': float(np.round(self.CLB, decimals)) if hasattr(self, 'CLB') else None,
            'ZL': float(np.round(self.ZL, decimals)),
            'Z0L': float(np.round(self.Z0L, decimals)),
            'Z0': float(np.round(self.Z0, decimals)),
            'Z0U': float(np.round(self.Z0U, decimals)),
            'ZU': float(np.round(self.ZU, decimals)),
            'CUB': float(np.round(self.CUB, decimals)) if hasattr(self, 'CUB') else None,
            'DUB': float(np.round(self.DUB, decimals)),
            'USB': float(np.round(self.USB, decimals)) if hasattr(self, 'USB') else None,
            'UB': float(np.round(self.UB, decimals)),
        }
    
    
    def _is_homogeneous(self):
        """
        Check if the data is homogeneous.
        Returns True if homogeneous, False otherwise.
        """
        self.ih = DataHomogeneity(gdf=self.init_egdf, 
                             catch=self.catch, 
                             verbose=self.verbose)
        is_homogeneous = self.ih.test_homogeneity(estimate_cluster_bounds=self.estimate_cluster_bounds) # NOTE set true as default because we want to get cluster bounds in marginal analysis
        # cluster bounds
        if self.estimate_cluster_bounds:
            self.CLB = self.ih.CLB
            self.CUB = self.ih.CUB

        if self.catch:
            self.params.update(self.ih.params)

        return is_homogeneous

    
    def _plot_egdf_intv_comprehensive(self, figsize=(12, 8)):
        """
        Generate comprehensive EGDF Interval Analysis plots with tolerance and typical data intervals.
        
        Creates visualization showing:
        - EGDF curve with distribution fitting
        - PDF curve on secondary axis
        - Tolerance interval (Z0L, Z0U) as light green filled zone
        - Typical data interval (ZL, ZU) as light blue filled zone
        - All critical points and bounds including Z0 vertical line
        - Original data points as rug plot
        
        Developer Notes:
        ----------------
        Creates visualization of both the interval analysis results and the
        underlying initial EGDF fit. Useful for debugging and result interpretation.
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 8)
            Figure size as (width, height) in inches
            
        Raises:
        -------
        RuntimeError : If called before fitting is complete
        """
        if not self._fitted:
            raise RuntimeError(
                "Cannot generate plots: Interval analysis not yet fitted. "
                "Please call the 'fit' method before attempting to plot results."
            )
        
        if not self.catch:
            print("Plot is not available with argument catch=False")
            return
            
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            raise ImportError("matplotlib and numpy required for plotting")
        
        # Create figure with dual y-axes
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = ax1.twinx()
        
        # Get data from fitted EGDF
        x_points = self.init_egdf.data
        egdf_vals = self.init_egdf.params.get('egdf') if hasattr(self.init_egdf, 'params') else None
        pdf_vals = self.init_egdf.params.get('pdf') if hasattr(self.init_egdf, 'params') else None
        wedf_vals = self.init_egdf.params.get('wedf') if hasattr(self.init_egdf, 'params') else None
        
        # Get smooth curve data if available
        smooth_x = None
        smooth_egdf = None
        smooth_pdf = None
        
        if hasattr(self.init_egdf, 'di_points_n') and self.init_egdf.di_points_n is not None:
            smooth_x = self.init_egdf.di_points_n
            
            if hasattr(self.init_egdf, 'egdf_points') and self.init_egdf.egdf_points is not None:
                smooth_egdf = self.init_egdf.egdf_points
                
            if hasattr(self.init_egdf, 'pdf_points') and self.init_egdf.pdf_points is not None:
                smooth_pdf = self.init_egdf.pdf_points
        
        # Set up x-axis range with padding
        if hasattr(self.init_egdf, 'DLB') and hasattr(self.init_egdf, 'DUB'):
            x_range = self.init_egdf.DUB - self.init_egdf.DLB
            x_pad = x_range * 0.05
            x_min = self.init_egdf.DLB - x_pad
            x_max = self.init_egdf.DUB + x_pad
        else:
            data_range = np.max(x_points) - np.min(x_points)
            x_pad = data_range * 0.05
            x_min = np.min(x_points) - x_pad
            x_max = np.max(x_points) + x_pad
        
        # Create fine x array for smooth interval zones
        x_fine = np.linspace(x_min, x_max, 1000)
        
        # ==================== PLOT EGDF CURVE ====================
        
        # Plot discrete EGDF points
        if egdf_vals is not None:
            ax1.plot(x_points, egdf_vals, 'o', color='blue', label='EGDF', markersize=4, alpha=0.7)
        
        # Plot smooth EGDF curve if available
        if smooth_x is not None and smooth_egdf is not None:
            ax1.plot(smooth_x, smooth_egdf, '-', color='blue', linewidth=2.5, alpha=0.9)
        
        # Plot WEDF if available
        if wedf_vals is not None:
            ax1.plot(x_points, wedf_vals, 's', color='lightblue', 
                    label='WEDF', markersize=3, alpha=0.6)
        
        # ==================== PLOT PDF CURVE ====================
        
        # Plot discrete PDF points
        if pdf_vals is not None:
            ax2.plot(x_points, pdf_vals, 'o', color='red', label='PDF', markersize=4, alpha=0.7)
            max_pdf = np.max(pdf_vals)
        else:
            max_pdf = 1.0
        
        # Plot smooth PDF curve if available
        if smooth_x is not None and smooth_pdf is not None:
            ax2.plot(smooth_x, smooth_pdf, '-', color='red', linewidth=2.5, alpha=0.9)
            max_pdf = max(max_pdf, np.max(smooth_pdf))
        
        # ==================== PLOT FILLED INTERVALS ====================
        
        # 1. Tolerance Interval (Z0L to Z0U) - Light Green
        if hasattr(self, 'intv') and self.intv and hasattr(self.intv, 'z0l') and hasattr(self.intv, 'z0u'):
            tolerance_mask = (x_fine >= self.intv.z0l) & (x_fine <= self.intv.z0u)
            
            if smooth_x is not None and smooth_egdf is not None:
                # Interpolate EGDF values for smooth filling
                tolerance_egdf = np.interp(x_fine[tolerance_mask], smooth_x, smooth_egdf)
                ax1.fill_between(x_fine[tolerance_mask], 0, tolerance_egdf, 
                               alpha=0.4, color='lightgreen', 
                               label=f'Tolerance Interval [Z0L: {self.intv.z0l:.3f}, Z0U: {self.intv.z0u:.3f}]')
            else:
                # Fallback to simple vertical fill
                ax1.axvspan(self.intv.z0l, self.intv.z0u, alpha=0.15, color='lightgreen',
                           label=f'Tolerance Interval [Z0L: {self.intv.z0l:.3f}, Z0U: {self.intv.z0u:.3f}]')
        
        # 2. Typical Data Interval (ZL to ZU) - Light Blue
        if hasattr(self, 'intv') and self.intv and hasattr(self.intv, 'zl') and hasattr(self.intv, 'zu'):
            typical_mask = (x_fine >= self.intv.zl) & (x_fine <= self.intv.zu)
            
            if smooth_x is not None and smooth_egdf is not None:
                # Interpolate EGDF values for smooth filling
                typical_egdf = np.interp(x_fine[typical_mask], smooth_x, smooth_egdf)
                ax1.fill_between(x_fine[typical_mask], 0, typical_egdf, 
                               alpha=0.1, color='blue',
                               label=f'Typical Data Interval [ZL: {self.intv.zl:.3f}, ZU: {self.intv.zu:.3f}]')
            else:
                # Fallback to simple vertical fill
                ax1.axvspan(self.intv.zl, self.intv.zu, alpha=0.15, color='lightblue',
                           label=f'Typical Data Interval [ZL: {self.intv.zl:.3f}, ZU: {self.intv.zu:.3f}]')
        
        # ==================== PLOT CRITICAL VERTICAL LINES ====================
        
        # ZL (lower datum) - Purple dashed
        if hasattr(self, 'intv') and self.intv and hasattr(self.intv, 'zl'):
            ax1.axvline(x=self.intv.zl, color='purple', linestyle='--', linewidth=2, 
                       alpha=0.8, label=f'ZL={self.intv.zl:.3f}')
        
        # Z0 (gnostic mode) - MAGENTA DASH-DOT (most prominent)
        z0_value = None
        if hasattr(self, 'intv') and self.intv and hasattr(self.intv, 'z0'):
            z0_value = self.intv.z0
        elif hasattr(self, 'intv') and self.intv and hasattr(self.intv, 'z0_original'):
            z0_value = self.intv.z0_original
        elif hasattr(self.init_egdf, 'z0'):
            z0_value = self.init_egdf.z0
        
        if z0_value is not None:
            ax1.axvline(x=z0_value, color='magenta', linestyle='-.', linewidth=1, 
                       alpha=0.9, label=f'Z0={z0_value:.3f}')
        
        # ZU (upper datum) - Orange dashed  
        if hasattr(self, 'intv') and self.intv and hasattr(self.intv, 'zu'):
            ax1.axvline(x=self.intv.zu, color='orange', linestyle='--', linewidth=2, 
                       alpha=0.8, label=f'ZU={self.intv.zu:.3f}')
        
        # ==================== PLOT DATA BOUNDS ====================
        
        # Data bounds (DLB, DUB) - Solid lines
        if hasattr(self.init_egdf, 'DLB') and self.init_egdf.DLB is not None:
            ax1.axvline(x=self.init_egdf.DLB, color='green', linestyle='-', 
                       linewidth=1, alpha=0.6, label=f'DLB={self.init_egdf.DLB:.3f}')
        if hasattr(self.init_egdf, 'DUB') and self.init_egdf.DUB is not None:
            ax1.axvline(x=self.init_egdf.DUB, color='orange', linestyle='-', 
                       linewidth=1, alpha=0.6, label=f'DUB={self.init_egdf.DUB:.3f}')
        
        # Probable bounds (LB, UB) - Dashed lines
        if hasattr(self.init_egdf, 'LB') and self.init_egdf.LB is not None:
            ax1.axvline(x=self.init_egdf.LB, color='purple', linestyle='--', 
                       linewidth=1, alpha=0.6, label=f'LB={self.init_egdf.LB:.3f}')
        if hasattr(self.init_egdf, 'UB') and self.init_egdf.UB is not None:
            ax1.axvline(x=self.init_egdf.UB, color='brown', linestyle='--', 
                       linewidth=1, alpha=0.6, label=f'UB={self.init_egdf.UB:.3f}')
        
        # Cluster bounds if available
        if hasattr(self, 'CLB') and self.CLB is not None:
            ax1.axvline(x=self.CLB, color='orange', linestyle='-.', 
                       linewidth=1, alpha=0.6, label=f'CLB={self.CLB:.3f}')
        if hasattr(self, 'CUB') and self.CUB is not None:
            ax1.axvline(x=self.CUB, color='orange', linestyle='-.', 
                       linewidth=1, alpha=0.6, label=f'CUB={self.CUB:.3f}')
        
        # Sample bounds if available
        if hasattr(self, 'LSB') and self.LSB is not None:
            ax1.axvline(x=self.LSB, color='gray', linestyle=':', 
                       linewidth=1, alpha=0.6, label=f'LSB={self.LSB:.3f}')
        if hasattr(self, 'USB') and self.USB is not None:
            ax1.axvline(x=self.USB, color='gray', linestyle=':', 
                       linewidth=1, alpha=0.6, label=f'USB={self.USB:.3f}')
        
        # ==================== PLOT DATA POINTS (RUG) - NO LEGEND ====================
        
        # Add original data points as rug plot at bottom (no legend entry)
        data_y_pos = -0.05  # Position below x-axis
        ax1.scatter(x_points, [data_y_pos] * len(x_points), 
                   alpha=0.6, s=15, color='black', marker='|')
        
        # ==================== FORMATTING ====================
        
        # Set axis labels and limits
        ax1.set_xlabel('Data Values', fontsize=12, fontweight='bold')
        ax1.set_ylabel('EGDF Value', fontsize=12, fontweight='bold', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_ylim(-0.1, 1.05)  # Extended to show rug plot
        ax1.set_xlim(x_min, x_max)
        
        ax2.set_ylabel('PDF Value', fontsize=12, fontweight='bold', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(0, max_pdf * 1.1)
        ax2.set_xlim(x_min, x_max)
        
        # Grid
        ax1.grid(True, alpha=0.3)
        
        # Build title with Z0 value
        title_text = 'EGDF Interval Analysis'
        if z0_value is not None:
            title_text += f' (Z0 = {z0_value:.3f})'
        
        plt.title(title_text, fontsize=14, fontweight='bold')
        
        # Combined legend - organized by category
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        
        # Filter and organize legend entries
        interval_entries = []
        critical_entries = []
        bound_entries = []
        curve_entries = []
        
        # Process ax1 entries
        for line, label in zip(lines1, labels1):
            if 'Interval' in label:
                interval_entries.append((line, label))
            elif any(marker in label for marker in ['ZL=', 'Z0=', 'ZU=']):
                critical_entries.append((line, label))
            elif any(bound in label for bound in ['LB=', 'UB=', 'DLB=', 'DUB=', 'CLB=', 'CUB=', 'LSB=', 'USB=']):
                bound_entries.append((line, label))
            elif any(curve in label for curve in ['EGDF', 'WEDF']):
                curve_entries.append((line, label))
        
        # Process ax2 entries
        for line, label in zip(lines2, labels2):
            if 'PDF' in label:
                curve_entries.append((line, label))
        
        # Combine in logical order: intervals, critical points, curves, bounds
        all_entries = interval_entries + critical_entries + curve_entries + bound_entries
        
        if all_entries:
            all_lines, all_labels = zip(*all_entries)
            ax1.legend(all_lines, all_labels, loc='upper left', fontsize=10, 
                      bbox_to_anchor=(0.02, 0.98))
        
        plt.tight_layout()
        plt.show()
        
        # Print summary information with Z0
        if self.verbose:
            print(f"\nEGDF Interval Analysis Plot Summary:")
            if hasattr(self, 'intv') and self.intv:
                if z0_value is not None:
                    print(f"  Z0 (Gnostic Mode): {z0_value:.4f}")
                if hasattr(self.intv, 'z0l') and hasattr(self.intv, 'z0u'):
                    tolerance_interval = self.intv.z0u - self.intv.z0l
                    print(f"  Tolerance interval: [{self.intv.z0l:.4f}, {self.intv.z0u:.4f}] (width: {tolerance_interval:.4f})")
                if hasattr(self.intv, 'zl') and hasattr(self.intv, 'zu'):
                    typical_data_interval = self.intv.zu - self.intv.zl
                    print(f"  Typical data interval: [{self.intv.zl:.4f}, {self.intv.zu:.4f}] (width: {typical_data_interval:.4f})")
                
                # Data coverage analysis
                if hasattr(self.intv, 'z0l') and hasattr(self.intv, 'z0u'):
                    data_in_tolerance = np.sum((x_points >= self.intv.z0l) & (x_points <= self.intv.z0u))
                    print(f"  Data coverage - Tolerance: {data_in_tolerance}/{len(x_points)} ({data_in_tolerance/len(x_points):.1%})")
                if hasattr(self.intv, 'zl') and hasattr(self.intv, 'zu'):
                    data_in_typical = np.sum((x_points >= self.intv.zl) & (x_points <= self.intv.zu))
                    print(f"  Data coverage - Typical: {data_in_typical}/{len(x_points)} ({data_in_typical/len(x_points):.1%})")
            
            print(f"  Total data points: {len(x_points)}")
            print(f"  Data range: [{np.min(x_points):.4f}, {np.max(x_points):.4f}]")
            
            # Homogeneity information
            if hasattr(self, 'h'):
                print(f"  Data homogeneity: {'Homogeneous' if self.h else 'Heterogeneous'}")
                
            # Clustering information if available
            if hasattr(self, 'main_cluster') and self.main_cluster is not None:
                print(f"  Main cluster size: {len(self.main_cluster)}")
            if hasattr(self, 'lower_cluster') and self.lower_cluster is not None:
                print(f"  Lower cluster size: {len(self.lower_cluster)}")
            if hasattr(self, 'upper_cluster') and self.upper_cluster is not None:
                print(f"  Upper cluster size: {len(self.upper_cluster)}")
    
    def _main_cluster_validation_and_msg(self):
        """
        Validates main cluster quality and provides user guidance.
        
        Developer Notes:
        ----------------
        Checks if the identified main cluster is suitable for reliable ELDF fitting.
        Issues warnings when cluster quality is insufficient, which could lead to
        unreliable interval estimates.
        
        Validation Criteria:
        - Main cluster exists (not None)  
        - Main cluster has at least 4 data points (minimum for ELDF fitting)
        
        Warning Conditions:
        - Cluster too small: < 4 points (insufficient for parameter estimation)
        - Cluster undefined: clustering algorithm failed to identify main group
        """
        # main cluster check
        if self.main_cluster is None or len(self.main_cluster) < 4:
            warnings.warn(
                "Insufficient main cluster data detected. "
                f"Main cluster has {len(self.main_cluster) if self.main_cluster is not None else 0} points, "
                "but at least 4 are required for reliable ELDF parameter estimation. "
                "This may result in unreliable interval estimates. "
                "Consider: (1) increasing data size, (2) adjusting cluster_threshold parameter, "
                "(3) setting homogeneous=True if outliers are not expected, or "
                "(4) reviewing data quality for potential issues.",
                UserWarning,
                stacklevel=2
            )
        else:
            if self.verbose:
                print(f"✓ Main cluster validated with {len(self.main_cluster)} data points.")

    def _get_global_cluster(self):
        """
        get data in range of LSB and USB if available, otherwise data as is

        main cluster is determined by LSB and USB if available
        lower cluster is data below LSB if available
        upper cluster is data above USB if available
        """
        if self.get_clusters:
            if self.verbose:
                print("Getting clusters from data...")
            if hasattr(self, 'LSB') and hasattr(self, 'USB') and self.LSB is not None and self.USB is not None:
                main_cluster = self.init_egdf.data[(self.init_egdf.data >= self.LSB) & (self.init_egdf.data <= self.USB)]
                lower_cluster = self.init_egdf.data[self.init_egdf.data < self.LSB]
                upper_cluster = self.init_egdf.data[self.init_egdf.data > self.USB]
            else:
                main_cluster = self.init_egdf.data
                lower_cluster = None
                upper_cluster = None

        return lower_cluster, main_cluster, upper_cluster

    def _fit_egdf_intv(self, plot=False):
        try:
            if self.verbose:
                print("\n\nFitting EGDF Interval Analysis...")
                
            # get initial EGDF
            self._get_initial_egdf(plot=plot)
            self.params = getattr(self.init_egdf, 'params', {}).copy()

            # homogeneous check
            self.h = self._is_homogeneous()

            if self.h:
                if self.verbose:
                    print("Data is homogeneous. Using homogeneous data for interval analysis.")
            else:
                if self.verbose:
                    print("Data is heterogeneous. Need to estimate cluster bounds to find main cluster. Recommend to use 'IntervalAnalysisELDF' instead.")

            # h check
            if self.h == False and self.estimate_cluster_bounds == False and self.get_clusters == True:
                warnings.warn("Data is heterogeneous but estimate_cluster_bounds is False. "
                            "Consider setting 'estimate_cluster_bounds=True' and 'get_clusters=True' to find main cluster bounds and main cluster. OR use 'IntervalAnalysisELDF' for better interval analysis results.")

            # optional data sampling bounds
            if self.estimate_sample_bounds:
                self._get_data_sample_bounds()

            # cluster bounds
            if self.estimate_cluster_bounds:
                self.lower_cluster, self.main_cluster, self.upper_cluster = self._get_cluster() # if get_clusters is True, it will estimate cluster bounds

            # main cluster check
            self._main_cluster_validation_and_msg()

            if self.verbose:
                print("Initiating EGDF Interval Analysis...")

            # intv analysis
            # get extended EGDF with a new datum
            self.intv = IntveEngine(self.init_egdf,
                                n_points_per_direction=self.n_points_per_direction,
                                dense_zone_fraction=self.dense_zone_fraction,
                                dense_points_fraction=self.dense_points_fraction,
                                convergence_window=self.convergence_window,
                                convergence_threshold=self.convergence_threshold,
                                min_search_points=self.min_search_points,
                                boundary_margin_factor=self.boundary_margin_factor,
                                extrema_search_tolerance=self.extrema_search_tolerance,
                                verbose=self.verbose)

            self.intv.fit(plot=plot, update_df_params=True)
            # extract results
            self.Z0 = self.intv.z0
            self.Z0L = self.intv.z0l
            self.Z0U = self.intv.z0u
            self.ZL = self.intv.zl
            self.ZU = self.intv.zu
            
            # fit status
            self._fitted = True

            # plot
            if plot:
                self._plot_egdf_intv_comprehensive()

            if self.verbose:
                print("EGDF Interval Analysis fitted successfully.")
        
        except Exception as e:
            if self.verbose:
                print(f"Error occurred during fitting: {e}")



    ############### DEPRECATED METHODS BELOW #################
    # def _compute_intv_scipy(self):
    #     '''
    #     using scipy minimize with constraints to find Z0L and Z0U with improved robustness
    #     '''
    #     if self.verbose:
    #         print("Computing interval values using scipy optimization with constraints...")
    
    #     try:
    #         from scipy.optimize import minimize
    #         import warnings
    #     except ImportError as e:
    #         raise ImportError("scipy is required for this method. Please install scipy.") from e
        
    #     # For ZL optimization: minimize Z0 subject to LB <= zl <= z0_main
    #     zol_bounds = [(self.LB, self._z0_main)]
        
    #     # For ZU optimization: maximize Z0 subject to z0_main <= zu <= UB  
    #     zou_bounds = [(self._z0_main, self.UB)]
    
    #     # Improved constraint functions with tolerance
    #     def constraint_z0l_value(x):
    #         """Ensure resulting Z0L <= z0_main"""
    #         try:
    #             z_egdf = self._create_extended_egdf_intv(x[0])
    #             z0_datum = self._get_z0(z_egdf)
    #             return self._z0_main - z0_datum + self.tolerance  # Add tolerance for numerical stability
    #         except Exception:
    #             return -1e6  # Large negative value if computation fails
    
    #     def constraint_z0u_value(x):
    #         """Ensure resulting Z0U >= z0_main"""
    #         try:
    #             z_egdf = self._create_extended_egdf_intv(x[0])
    #             z0_datum = self._get_z0(z_egdf)
    #             return z0_datum - self._z0_main + self.tolerance  # Add tolerance for numerical stability
    #         except Exception:
    #             return -1e6  # Large negative value if computation fails
    
    #     # Define constraints for scipy
    #     constraints_zol = [
    #         {'type': 'ineq', 'fun': constraint_z0l_value}
    #     ]
        
    #     constraints_zou = [
    #         {'type': 'ineq', 'fun': constraint_z0u_value}
    #     ]
    
    #     # Robust objective functions with error handling
    #     def objective_zol(datum):
    #         try:
    #             z_egdf = self._create_extended_egdf_intv(datum[0])
    #             z0_datum = self._get_z0(z_egdf)
    #             return z0_datum
    #         except Exception:
    #             return 1e6  # Large positive value if computation fails
        
    #     def objective_zou(datum):
    #         try:
    #             z_egdf = self._create_extended_egdf_intv(datum[0])
    #             z0_datum = self._get_z0(z_egdf)
    #             return -z0_datum  # negative because we want to maximize Z0
    #         except Exception:
    #             return 1e6  # Large positive value if computation fails
    
    #     # Multiple optimization attempts with different initial points and methods
    #     methods = ['SLSQP', 'trust-constr']
        
    #     # Initialize with fallback values
    #     self.zl = float(self._z0_main)
    #     self.z0l = float(self._z0_main)
    #     self.zu = float(self._z0_main)
    #     self.z0u = float(self._z0_main)
        
    #     zol_success = False
    #     zou_success = False
        
    #     for method in methods:
    #         if zol_success and zou_success:
    #             break
                
    #         # Try different initial points for ZL optimization
    #         if not zol_success:
    #             initial_points_zol = [
    #                 [self._z0_main],
    #                 [self.LB + 0.1 * (self._z0_main - self.LB)],
    #                 [self.LB + 0.5 * (self._z0_main - self.LB)],
    #                 [self.LB + 0.9 * (self._z0_main - self.LB)]
    #             ]
                
    #             for x0_zol in initial_points_zol:
    #                 try:
    #                     res_zol = minimize(
    #                         objective_zol, 
    #                         x0=np.array(x0_zol), 
    #                         method=method,
    #                         bounds=zol_bounds,
    #                         constraints=constraints_zol,
    #                         options={'ftol': max(self.tolerance, 1e-12), 'maxiter': 1000}
    #                     )
                        
    #                     if res_zol.success and res_zol.fun <= self._z0_main + self.tolerance:
    #                         self.zl = float(res_zol.x[0])
    #                         self.z0l = float(res_zol.fun)
    #                         zol_success = True
    #                         break
    #                 except Exception:
    #                     continue
            
    #         # Try different initial points for ZU optimization
    #         if not zou_success:
    #             initial_points_zou = [
    #                 [self._z0_main],
    #                 [self._z0_main + 0.1 * (self.UB - self._z0_main)],
    #                 [self._z0_main + 0.5 * (self.UB - self._z0_main)],
    #                 [self._z0_main + 0.9 * (self.UB - self._z0_main)]
    #             ]
                
    #             for x0_zou in initial_points_zou:
    #                 try:
    #                     res_zou = minimize(
    #                         objective_zou,
    #                         x0=np.array(x0_zou), 
    #                         method=method,
    #                         bounds=zou_bounds,
    #                         constraints=constraints_zou,
    #                         options={'ftol': max(self.tolerance, 1e-12), 'maxiter': 1000}
    #                     )
                        
    #                     if res_zou.success and (-res_zou.fun) >= self._z0_main - self.tolerance:
    #                         self.zu = float(res_zou.x[0])
    #                         self.z0u = float(-res_zou.fun)
    #                         zou_success = True
    #                         break
    #                 except Exception:
    #                     continue
    
    #     # Post-processing to ensure ordering
    #     if self.z0l > self._z0_main:
    #         self.z0l = float(self._z0_main)
    #         zol_success = False
        
    #     if self.z0u < self._z0_main:
    #         self.z0u = float(self._z0_main)
    #         zou_success = False
        
    #     # Ensure datum ordering by adjusting if necessary
    #     if self.zl > self._z0_main:
    #         self.zl = float(self._z0_main)
        
    #     if self.zu < self._z0_main:
    #         self.zu = float(self._z0_main)
        
    #     # Set Z0
    #     self.z0 = float(self._z0_main)
    
    #     # Final validation and correction
    #     ordering_satisfied = (self.zl <= self.z0l <= self.z0 <= self.z0u <= self.zu)
        
    #     if not ordering_satisfied:
    #         warnings.warn("Ordering constraint violated. Applying corrections...")
            
    #         # Apply minimum corrections to satisfy ordering
    #         if self.zl > self.z0l:
    #             self.zl = self.z0l
    #         if self.z0l > self.z0:
    #             self.z0l = self.z0
    #         if self.z0 > self.z0u:
    #             self.z0u = self.z0
    #         if self.z0u > self.zu:
    #             self.zu = self.z0u
            
    #         ordering_satisfied = (self.zl <= self.z0l <= self.z0 <= self.z0u <= self.zu)
        
    #     if self.verbose:
    #         print(f"\nInterval Analysis Results (Scipy Optimization with Constraints):")
    #         print(f"ZL:  {self.zl:.6f}")
    #         print(f"Z0L: {self.z0l:.6f}")
    #         print(f"Z0:  {self.z0:.6f}")
    #         print(f"Z0U: {self.z0u:.6f}")
    #         print(f"ZU:  {self.zu:.6f}")
    #         print(f"Ordering constraint satisfied: {ordering_satisfied}")
    #         print(f"ZL optimization: {'Success' if zol_success else 'Failed'}")
    #         print(f"ZU optimization: {'Success' if zou_success else 'Failed'}")
    
    #     if self.catch:
    #         self.params = getattr(self.init_egdf, 'params', {}).copy()
    #         self.params.update({
    #             'ZL': self.zl,
    #             'Z0L': self.z0l,
    #             'Z0': float(self.z0),
    #             'Z0U': self.z0u,
    #             'ZU': self.zu,
    #             'optimization_success': zol_success and zou_success,
    #             'ordering_satisfied': ordering_satisfied
    #         })
    
    # def _compute_intv(self):
    #     '''
    #     first compute interval using scipy minimize
    #     if it fails then use linear search method
    #     '''
    #     # NOTE
    #     # two different methods are used for interval computation _get_z0 and _get_z0_main
    #     # _get_z0_main is more robust and used for main computations of g-mode of the sample, this method ensure that egdf is close to 0.5, hence high accuracy with high iterations
    #     # _get_z0 is faster and used for interval computations, this method may not be as accurate as _get_z0_main, but is faster. Main difference is that, there is no penalty logic of egdf being not close to 0.5.

    #     # _compute_intv_scipy uses _get_z0_main for better robustness
    #     # _compute_intv_linear_search uses _get_z0 for faster computation

    #     if self.verbose:
    #         print("Initiating interval computation...")

    #     # compute with fallback
    #     try:
    #         self._compute_intv_scipy()
    #     except Exception as e:
    #         warnings.warn(f"Scipy optimization failed: {e}. Falling back to linear search method.")
    #         self._compute_intv_linear_search()


    # def _compute_intv_linear_search(self): # NOTE in future, this computation may be optimized
    #     """
    #     Compute interval values including Z0L, Z0U, ZL, and ZU.
        
    #     This method calculates critical interval boundaries based on the Z0 interval
    #     and the central point Z0. It identifies tolerance bounds and typical value ranges.
    #     Enhanced with convergence detection for plateau behavior and improved logic.
    #     """
    #     try:
    #         # Input validation
    #         if not hasattr(self, 'z0') or self.z0 is None:
    #             raise ValueError("Z0 must be computed before interval analysis. Call _get_z0() first.")
            
    #         if not hasattr(self, 'LB') or not hasattr(self, 'UB') or self.LB is None or self.UB is None:
    #             raise ValueError("LB and UB must be defined for interval computation.")
            
    #         if self.LB >= self.UB:
    #             raise ValueError(f"Lower bound ({self.LB}) must be less than upper bound ({self.UB}).")
            
    #         if not (self.LB <= self.z0 <= self.UB):
    #             warnings.warn(f"Z0 ({self.z0}) is outside the bounds [{self.LB}, {self.UB}].")
            
    #         # Initialize empty lists and variables
    #         self.z0_interval = []
    #         self.datum_range = []
            
    #         # Initialize interval tracking with separate lists for lower and upper searches
    #         lower_search_data = {'datum': [], 'z0': []}
    #         upper_search_data = {'datum': [], 'z0': []}
            
    #         if self.verbose:
    #             print("Computing interval values...")
    #             print(f"Z0: {self.z0:.6f}, LB: {self.LB:.6f}, UB: {self.UB:.6f}")
            
    #         # Ensure we have enough points for meaningful analysis
    #         min_points_per_side = max(100, self.n_points)
    #         points_per_side = max(min_points_per_side, self.n_points)
            
    #         # Initialize tracking variables
    #         current_z0_min = self.z0
    #         current_z0_max = self.z0
    #         z0l_datum = self.z0
    #         z0u_datum = self.z0
            
    #         # Early stopping parameters
    #         early_stop_tolerance = self._TOLERANCE
    #         consecutive_increases = 0
    #         consecutive_decreases = 0
    #         max_consecutive = 5
            
    #         # Convergence/Plateau detection parameters
    #         convergence_tolerance = self._TOLERANCE
    #         plateau_window = 5
    #         min_plateau_points = 5
            
    #         # Search towards lower bound
    #         if self._z0_main > self.LB:
    #             lower_range = np.linspace(self._z0_main, self.LB, points_per_side)
    #             z0_history_lower = []
    #             lower_z0_min = self._z0_main
    #             lower_z0l_datum = self._z0_main

    #             for i, datum in enumerate(lower_range[1:], 1):
    #                 try:
    #                     if self.verbose and i % max(1, points_per_side) == 0:
    #                         print(f"Processing lower range: {datum:.6f} ({i}/{len(lower_range)-1})")
                        
    #                     z_egdf = self._create_extended_egdf_intv(datum)
    #                     z0_datum = self._get_z0(z_egdf)
                        
    #                     # Store in separate lower search data
    #                     lower_search_data['datum'].append(datum)
    #                     lower_search_data['z0'].append(z0_datum)
                        
    #                     # Also store in combined arrays for backward compatibility
    #                     self.z0_interval.append(z0_datum)
    #                     self.datum_range.append(datum)
    #                     z0_history_lower.append(z0_datum)
                        
    #                     # Track minimum Z0 in lower search specifically
    #                     if z0_datum < lower_z0_min:
    #                         lower_z0_min = z0_datum
    #                         lower_z0l_datum = datum
    #                         consecutive_increases = 0
    #                     else:
    #                         consecutive_increases += 1
                        
    #                     # Update global minimum
    #                     if z0_datum < current_z0_min:
    #                         current_z0_min = z0_datum
                        
    #                     # Convergence/Plateau detection (same as before)
    #                     if len(z0_history_lower) >= min_plateau_points:
    #                         recent_window = min(plateau_window, len(z0_history_lower))
    #                         recent_z0_values = z0_history_lower[-recent_window:]
                            
    #                         z0_variance = np.var(recent_z0_values)
    #                         z0_range = np.max(recent_z0_values) - np.min(recent_z0_values)
                            
    #                         if (z0_variance < convergence_tolerance and z0_range < convergence_tolerance):
    #                             if self.verbose:
    #                                 print(f"Convergence detected at lower bound: Z0 plateau reached")
    #                             break
                            
    #                         mean_recent_z0 = np.mean(recent_z0_values)
    #                         max_deviation = np.max(np.abs(recent_z0_values - mean_recent_z0))
                            
    #                         if max_deviation < convergence_tolerance:
    #                             if self.verbose:
    #                                 print(f"Convergence detected at lower bound: Z0 stabilized")
    #                             break
                        
    #                     # Early stopping logic
    #                     if (z0_datum > lower_z0_min + early_stop_tolerance and 
    #                         consecutive_increases >= max_consecutive):
    #                         if self.verbose:
    #                             print(f"Early stopping at lower bound: Z0 increasing for {consecutive_increases} consecutive points")
    #                         break
                            
    #                 except Exception as e:
    #                     warnings.warn(f"Error processing datum {datum:.6f} in lower range: {e}")
    #                     continue
            
    #         # Search towards upper bound
    #         if self._z0_main < self.UB:
    #             upper_range = np.linspace(self._z0_main, self.UB, points_per_side)
    #             z0_history_upper = []
    #             upper_z0_max = self._z0_main
    #             upper_z0u_datum = self._z0_main

    #             for i, datum in enumerate(upper_range[1:], 1):
    #                 try:
    #                     if self.verbose and i % max(1, points_per_side) == 0:
    #                         print(f"Processing upper range: {datum:.6f} ({i}/{len(upper_range)-1})")
                        
    #                     z_egdf = self._create_extended_egdf_intv(datum)
    #                     z0_datum = self._get_z0(z_egdf)
                        
    #                     # Store in separate upper search data
    #                     upper_search_data['datum'].append(datum)
    #                     upper_search_data['z0'].append(z0_datum)
                        
    #                     # Also store in combined arrays for backward compatibility
    #                     self.z0_interval.append(z0_datum)
    #                     self.datum_range.append(datum)
    #                     z0_history_upper.append(z0_datum)
                        
    #                     # Track maximum Z0 in upper search specifically
    #                     if z0_datum > upper_z0_max:
    #                         upper_z0_max = z0_datum
    #                         upper_z0u_datum = datum
    #                         consecutive_decreases = 0
    #                     else:
    #                         consecutive_decreases += 1
                        
    #                     # Update global maximum
    #                     if z0_datum > current_z0_max:
    #                         current_z0_max = z0_datum
                        
    #                     # Convergence/Plateau detection (same as before)
    #                     if len(z0_history_upper) >= min_plateau_points:
    #                         recent_window = min(plateau_window, len(z0_history_upper))
    #                         recent_z0_values = z0_history_upper[-recent_window:]
                            
    #                         z0_variance = np.var(recent_z0_values)
    #                         z0_range = np.max(recent_z0_values) - np.min(recent_z0_values)
                            
    #                         if (z0_variance < convergence_tolerance and z0_range < convergence_tolerance):
    #                             if self.verbose:
    #                                 print(f"Convergence detected at upper bound: Z0 plateau reached")
    #                             break
                            
    #                         mean_recent_z0 = np.mean(recent_z0_values)
    #                         max_deviation = np.max(np.abs(recent_z0_values - mean_recent_z0))
                            
    #                         if max_deviation < convergence_tolerance:
    #                             if self.verbose:
    #                                 print(f"Convergence detected at upper bound: Z0 stabilized")
    #                             break
                        
    #                     # Early stopping logic
    #                     if (z0_datum < upper_z0_max - early_stop_tolerance and 
    #                         consecutive_decreases >= max_consecutive):
    #                         if self.verbose:
    #                             print(f"Early stopping at upper bound: Z0 decreasing for {consecutive_decreases} consecutive points")
    #                         break
                            
    #                 except Exception as e:
    #                     warnings.warn(f"Error processing datum {datum:.6f} in upper range: {e}")
    #                     continue
            
    #         # Add Z0 point if not already included
    #         if self.z0 not in self.datum_range:
    #             self.z0_interval.append(self.z0)
    #             self.datum_range.append(self.z0)
            
    #         # Validate we have enough data points
    #         if len(self.z0_interval) < 3:
    #             warnings.warn("Insufficient data points for reliable interval analysis.")
            
    #         # Convert to numpy arrays
    #         self.z0_interval = np.array(self.z0_interval)
    #         self.datum_range = np.array(self.datum_range)
            
    #         # Remove any NaN or infinite values
    #         valid_mask = np.isfinite(self.z0_interval) & np.isfinite(self.datum_range)
    #         if not np.all(valid_mask):
    #             warnings.warn(f"Removing {np.sum(~valid_mask)} invalid data points from interval analysis.")
    #             self.z0_interval = self.z0_interval[valid_mask]
    #             self.datum_range = self.datum_range[valid_mask]
            
    #         if len(self.z0_interval) == 0:
    #             raise ValueError("No valid data points remaining after cleaning.")
            
    #         # IMPROVED LOGIC: Use direction-specific results instead of global min/max
    #         # Find Z0L and Z0U from respective search directions
    #         self.z0 = self._z0_main
            
    #         if len(lower_search_data['z0']) > 0:
    #             lower_z0_array = np.array(lower_search_data['z0'])
    #             lower_datum_array = np.array(lower_search_data['datum'])
    #             z0l_idx = np.argmin(lower_z0_array)
    #             self.z0l = float(lower_z0_array[z0l_idx])
    #             self.zl = float(lower_datum_array[z0l_idx])
    #         else:
    #             self.z0l = float(self.z0)
    #             self.zl = float(self.z0)
            
    #         if len(upper_search_data['z0']) > 0:
    #             upper_z0_array = np.array(upper_search_data['z0'])
    #             upper_datum_array = np.array(upper_search_data['datum'])
    #             z0u_idx = np.argmax(upper_z0_array)
    #             self.z0u = float(upper_z0_array[z0u_idx])
    #             self.zu = float(upper_datum_array[z0u_idx])
    #         else:
    #             self.z0u = float(self.z0)
    #             self.zu = float(self.z0)
            
    #         # Ensure logical ordering: ZL ≤ Z0 ≤ ZU
    #         if not (self.zl <= self.z0 <= self.zu):
    #             if self.verbose:
    #                 print(f"Swapping ZL and ZU: ZL was {self.zl:.6f}, ZU was {self.zu:.6f}")
    #             if self.zl > self.z0l:
    #                 self.zl = self.z0l
    #             if self.zu < self.z0u:
    #                 self.zu = self.z0u
    #         if not (self.z0l <= self.z0 <= self.z0u):
    #             if self.verbose:
    #                 print(f"Swapping Z0L and Z0U: Z0L was {self.z0l:.6f}, Z0U was {self.z0u:.6f}")
    #             if self.z0l > self.z0:
    #                 self.z0l = self.z0
    #             if self.z0u < self.z0:
    #                 self.z0u = self.z0
            
    #         # Additional validation
    #         if self.z0l > self.z0u:
    #             if self.verbose:
    #                 print(f"Warning: Z0L ({self.z0l:.6f}) > Z0U ({self.z0u:.6f}). Using fallback logic.")
    #             # Fallback: use global min/max from combined data
    #             z0l_idx_global = np.argmin(self.z0_interval)
    #             z0u_idx_global = np.argmax(self.z0_interval)
    #             self.z0l = float(self.z0_interval[z0l_idx_global])
    #             self.z0u = float(self.z0_interval[z0u_idx_global])
    #             self.zl = float(self.datum_range[z0l_idx_global])
    #             self.zu = float(self.datum_range[z0u_idx_global])
            
    #         # Store parameters if catching is enabled
    #         if self.catch:
    #             self.params = getattr(self.init_egdf, 'params', {}).copy()
    #             self.params.update({
    #                 'ZL': self.zl,
    #                 'Z0L': self.z0l,
    #                 'Z0': float(self.z0),
    #                 'Z0U': self.z0u,
    #                 'ZU': self.zu,
    #                 'lower_search_points': len(lower_search_data['z0']),
    #                 'upper_search_points': len(upper_search_data['z0'])
    #             })
            
    #         # Verbose output
    #         if self.verbose:
    #             print(f"\nInterval Analysis Results:")
    #             print(f"ZL:  {self.zl:.6f}")
    #             print(f"Z0L: {self.z0l:.6f}")
    #             print(f"Z0:  {self.z0:.6f}")
    #             print(f"Z0U: {self.z0u:.6f}")
    #             print(f"ZU:  {self.zu:.6f}")
    #             print(f"Lower search points: {len(lower_search_data['z0'])}")
    #             print(f"Upper search points: {len(upper_search_data['z0'])}")
    #             print("Interval values computed successfully.")
        
    #     except Exception as e:
    #         error_msg = f"Error in interval computation: {e}"
    #         if self.verbose:
    #             print(error_msg)
    #         raise RuntimeError(error_msg) from e