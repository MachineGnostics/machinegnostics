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
                estimate_sample_bounds: bool = False,
                estimate_cluster_bounds: bool = False,
                sample_bound_tolerance: float = 0.1,
                max_iterations: int = 10000,
                early_stopping_steps: int = 10,
                estimating_rate: float = 0.1, # NOTE for intv specific
                cluster_threshold: float = 0.05,
                get_clusters: bool = True,
                DLB: float = None,
                DUB: float = None,
                LB: float = None,
                UB: float = None,
                S = 'auto',
                tolerance: float = 1e-6,
                data_form: str = 'a',
                n_points: int = 1000, # NOTE for intv specific
                homogeneous: bool = True,
                catch: bool = True,
                weights: np.ndarray = None,
                wedf: bool = True,
                opt_method: str = 'L-BFGS-B',
                verbose: bool = False,
                max_data_size: int = 1000,
                flush: bool = True): # NOTE for intv specific
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

        # input validation
        if not isinstance(self.estimate_sample_bounds, bool):
            raise ValueError("estimate_sample_bounds must be a boolean.")
        if not isinstance(self.estimate_cluster_bounds, bool):
            raise ValueError("estimate_cluster_bounds must be a boolean.")

    def _create_extended_egdf_intv(self, datum):
        """Create EGDF with extended data including the given datum."""
        data_extended = np.append(self.init_egdf.data, datum)
        
        egdf_extended = EGDF(
            data=data_extended,
            tolerance=self.tolerance,
            data_form=self.data_form,
            n_points=self.n_points,
            homogeneous=self.homogeneous,
            S=self.S_opt,
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
        
        egdf_extended.fit()
        return egdf_extended

    def _compute_intv(self):
        """
        Compute interval values including Z0L, Z0U, ZL, and ZU.
        
        This method calculates critical interval boundaries based on the Z0 interval
        and the central point Z0. It identifies tolerance bounds and typical value ranges.
        Enhanced with convergence detection for plateau behavior.
        """
        try:
            # Input validation
            if not hasattr(self, 'z0') or self.z0 is None:
                raise ValueError("Z0 must be computed before interval analysis. Call _get_z0() first.")
            
            if not hasattr(self, 'LB') or not hasattr(self, 'UB') or self.LB is None or self.UB is None:
                raise ValueError("LB and UB must be defined for interval computation.")
            
            if self.LB >= self.UB:
                raise ValueError(f"Lower bound ({self.LB}) must be less than upper bound ({self.UB}).")
            
            if not (self.LB <= self.z0 <= self.UB):
                warnings.warn(f"Z0 ({self.z0}) is outside the bounds [{self.LB}, {self.UB}].")
            
            # Initialize empty lists and variables
            self.z0_interval = []
            self.datum_range = []
            
            if self.verbose:
                print("Computing interval values...")
                print(f"Z0: {self.z0:.6f}, LB: {self.LB:.6f}, UB: {self.UB:.6f}")
            
            # Ensure we have enough points for meaningful analysis
            min_points_per_side = max(10, self.n_points // 100)
            points_per_side = max(min_points_per_side, self.n_points // 2)
            
            # Initialize tracking variables
            current_z0_min = self.z0
            current_z0_max = self.z0
            z0l_datum = self.z0
            z0u_datum = self.z0
            
            # Early stopping parameters NOTE
            early_stop_tolerance = 1e-6  # Tolerance for early stopping
            consecutive_increases = 0
            consecutive_decreases = 0
            max_consecutive = 5
            
            # Convergence/Plateau detection parameters NOTE
            convergence_tolerance = 1e-6  # Tolerance for Z0 change detection
            plateau_window = 5  # Number of consecutive points to check for plateau
            min_plateau_points = 3  # Minimum points required before checking for plateau
            
            # Search towards lower bound
            if self.z0 > self.LB:
                lower_range = np.linspace(self.z0, self.LB, points_per_side)
                z0_history_lower = []  # Track recent Z0 values for plateau detection
    
                for i, datum in enumerate(lower_range[1:], 1):  # Skip first point (z0)
                    try:
                        if self.verbose and i % max(1, points_per_side // 10) == 0:
                            print(f"Processing lower range: {datum:.6f} ({i}/{len(lower_range)-1})")
                        
                        z_egdf = self._create_extended_egdf_intv(datum)
                        z0_datum = self._get_z0(z_egdf)
                        
                        self.z0_interval.append(z0_datum)
                        self.datum_range.append(datum)
                        z0_history_lower.append(z0_datum)
                        
                        # Check if we found a new minimum
                        if z0_datum < current_z0_min:
                            current_z0_min = z0_datum
                            z0l_datum = datum
                            consecutive_increases = 0
                        else:
                            consecutive_increases += 1
                        
                        # Convergence/Plateau detection
                        if len(z0_history_lower) >= min_plateau_points:
                            # Check if we have enough points to evaluate plateau
                            recent_window = min(plateau_window, len(z0_history_lower))
                            recent_z0_values = z0_history_lower[-recent_window:]
                            
                            # Calculate variance and range of recent Z0 values
                            z0_variance = np.var(recent_z0_values)
                            z0_range = np.max(recent_z0_values) - np.min(recent_z0_values)
                            
                            # Check for plateau (low variance and small range)
                            if (z0_variance < convergence_tolerance and 
                                z0_range < convergence_tolerance):
                                if self.verbose:
                                    print(f"Convergence detected at lower bound: Z0 plateau reached")
                                    print(f"  Variance: {z0_variance:.8f}, Range: {z0_range:.8f}")
                                    print(f"  Recent Z0 values: {[f'{z:.6f}' for z in recent_z0_values]}")
                                break
                            
                            # Alternative plateau detection: check if all recent values are within tolerance
                            mean_recent_z0 = np.mean(recent_z0_values)
                            max_deviation = np.max(np.abs(recent_z0_values - mean_recent_z0))
                            
                            if max_deviation < convergence_tolerance:
                                if self.verbose:
                                    print(f"Convergence detected at lower bound: Z0 stabilized")
                                    print(f"  Max deviation from mean: {max_deviation:.8f}")
                                    print(f"  Recent Z0 values: {[f'{z:.6f}' for z in recent_z0_values]}")
                                break
                        
                        # Original early stopping logic (kept as backup)
                        if (z0_datum > current_z0_min + early_stop_tolerance and 
                            consecutive_increases >= max_consecutive):
                            if self.verbose:
                                print(f"Early stopping at lower bound: Z0 increasing for {consecutive_increases} consecutive points")
                            break
                            
                    except Exception as e:
                        warnings.warn(f"Error processing datum {datum:.6f} in lower range: {e}")
                        continue
            
            # Search towards upper bound
            if self.z0 < self.UB:
                upper_range = np.linspace(self.z0, self.UB, points_per_side)
                z0_history_upper = []  # Track recent Z0 values for plateau detection
    
                for i, datum in enumerate(upper_range[1:], 1):  # Skip first point (z0)
                    try:
                        if self.verbose and i % max(1, points_per_side // 10) == 0:
                            print(f"Processing upper range: {datum:.6f} ({i}/{len(upper_range)-1})")
                        
                        z_egdf = self._create_extended_egdf_intv(datum)
                        z0_datum = self._get_z0(z_egdf)
                        
                        self.z0_interval.append(z0_datum)
                        self.datum_range.append(datum)
                        z0_history_upper.append(z0_datum)
                        
                        # Check if we found a new maximum
                        if z0_datum > current_z0_max:
                            current_z0_max = z0_datum
                            z0u_datum = datum
                            consecutive_decreases = 0
                        else:
                            consecutive_decreases += 1
                        
                        # Convergence/Plateau detection
                        if len(z0_history_upper) >= min_plateau_points:
                            # Check if we have enough points to evaluate plateau
                            recent_window = min(plateau_window, len(z0_history_upper))
                            recent_z0_values = z0_history_upper[-recent_window:]
                            
                            # Calculate variance and range of recent Z0 values
                            z0_variance = np.var(recent_z0_values)
                            z0_range = np.max(recent_z0_values) - np.min(recent_z0_values)
                            
                            # Check for plateau (low variance and small range)
                            if (z0_variance < convergence_tolerance and 
                                z0_range < convergence_tolerance):
                                if self.verbose:
                                    print(f"Convergence detected at upper bound: Z0 plateau reached")
                                    print(f"  Variance: {z0_variance:.8f}, Range: {z0_range:.8f}")
                                    print(f"  Recent Z0 values: {[f'{z:.6f}' for z in recent_z0_values]}")
                                break
                            
                            # Alternative plateau detection: check if all recent values are within tolerance
                            mean_recent_z0 = np.mean(recent_z0_values)
                            max_deviation = np.max(np.abs(recent_z0_values - mean_recent_z0))
                            
                            if max_deviation < convergence_tolerance:
                                if self.verbose:
                                    print(f"Convergence detected at upper bound: Z0 stabilized")
                                    print(f"  Max deviation from mean: {max_deviation:.8f}")
                                    print(f"  Recent Z0 values: {[f'{z:.6f}' for z in recent_z0_values]}")
                                break
                        
                        # Original early stopping logic (kept as backup)
                        if (z0_datum < current_z0_max - early_stop_tolerance and 
                            consecutive_decreases >= max_consecutive):
                            if self.verbose:
                                print(f"Early stopping at upper bound: Z0 decreasing for {consecutive_decreases} consecutive points")
                            break
                            
                    except Exception as e:
                        warnings.warn(f"Error processing datum {datum:.6f} in upper range: {e}")
                        continue
            
            # Add Z0 point if not already included
            if self.z0 not in self.datum_range:
                self.z0_interval.append(self.z0)
                self.datum_range.append(self.z0)
            
            # Validate we have enough data points
            if len(self.z0_interval) < 3:
                warnings.warn("Insufficient data points for reliable interval analysis. Consider increasing n_points or adjusting bounds.")
            
            # Convert to numpy arrays for easier manipulation
            self.z0_interval = np.array(self.z0_interval)
            self.datum_range = np.array(self.datum_range)
            
            # Remove any NaN or infinite values
            valid_mask = np.isfinite(self.z0_interval) & np.isfinite(self.datum_range)
            if not np.all(valid_mask):
                warnings.warn(f"Removing {np.sum(~valid_mask)} invalid data points from interval analysis.")
                self.z0_interval = self.z0_interval[valid_mask]
                self.datum_range = self.datum_range[valid_mask]
            
            if len(self.z0_interval) == 0:
                raise ValueError("No valid data points remaining after cleaning.")
            
            # Find the indices of minimum and maximum Z0 values in the interval
            z0l_idx = np.argmin(self.z0_interval)
            z0u_idx = np.argmax(self.z0_interval)
            
            # Set the interval values
            self.z0l = float(self.z0_interval[z0l_idx])
            self.z0u = float(self.z0_interval[z0u_idx])
            self.zl = float(self.datum_range[z0l_idx])
            self.zu = float(self.datum_range[z0u_idx])
            
            # Additional validation of computed values
            if self.z0l > self.z0u:
                warnings.warn(f"Z0L ({self.z0l:.6f}) > Z0U ({self.z0u:.6f}). This may indicate computational issues.")
            
            if self.zl > self.zu:
                warnings.warn(f"ZL ({self.zl:.6f}) > ZU ({self.zu:.6f}). This may indicate computational issues.")
            
            # Store parameters if catching is enabled
            if self.catch:
                self.params = getattr(self.init_egdf, 'params', {}).copy()
                self.params.update({
                    'ZL': self.zl,
                    'Z0L': self.z0l,
                    'Z0': float(self.z0),
                    'Z0U': self.z0u,
                    'ZU': self.zu,
                    # 'n_interval_points': len(self.z0_interval),
                    # 'convergence_tolerance': convergence_tolerance,
                    # 'plateau_window': plateau_window
                })
            
            # Verbose output
            if self.verbose:
                print(f"\nInterval Analysis Results:")
                print(f"ZL:  {self.zl:.6f} (datum producing minimum Z0)")
                print(f"Z0L: {self.z0l:.6f} (minimum Z0 value)")
                print(f"Z0:  {self.z0:.6f} (original central point)")
                print(f"Z0U: {self.z0u:.6f} (maximum Z0 value)")
                print(f"ZU:  {self.zu:.6f} (datum producing maximum Z0)")
                # print(f"Total points analyzed: {len(self.z0_interval)}")
                print("Interval values computed successfully.")
        
        except Exception as e:
            error_msg = f"Error in interval computation: {e}"
            if self.verbose:
                print(error_msg)
            raise RuntimeError(error_msg) from e

    
    def _plot_egdf_intv(self, plot_style='scatter', show_data_points=True, show_grid=True, figsize=(12, 8)):
        """
        Plot interval analysis results showing data range vs Z0 intervals.
        
        Parameters:
        -----------
        plot_style : str, default='smooth'
            Style of the main curve. Options: 'smooth', 'scatter', 'line'
        show_data_points : bool, default=True
            Whether to show individual data points as scatter
        show_grid : bool, default=True
            Whether to show grid lines
        figsize : tuple, default=(12, 8)
            Figure size as (width, height)
        
        Creates a comprehensive visualization with:
        - X-axis: Data range (datum values) with proper limits
        - Y-axis: Z0 interval values
        - Logical legend order: ZL - Z0L - Z0 - Z0U - ZU
        - Robust error handling and fallback options
        """
        try:
            import matplotlib.pyplot as plt
            
            # Check if interval computation has been performed
            if not hasattr(self, 'z0_interval') or not hasattr(self, 'datum_range'):
                raise ValueError("Intervals have not been computed yet. Run _compute_intv() first.")
            
            if len(self.z0_interval) == 0 or len(self.datum_range) == 0:
                raise ValueError("No data available for plotting. Interval computation may have failed.")
            
            # Validate required attributes
            required_attrs = ['zl', 'z0l', 'z0', 'z0u', 'zu']
            missing_attrs = [attr for attr in required_attrs if not hasattr(self, attr)]
            if missing_attrs:
                raise ValueError(f"Missing required attributes for plotting: {missing_attrs}")
            
            # Validate plot_style
            valid_styles = ['smooth', 'scatter', 'line']
            if plot_style not in valid_styles:
                warnings.warn(f"Invalid plot_style '{plot_style}'. Using 'smooth' instead.")
                plot_style = 'smooth'
            
            # Create the plot with error handling
            try:
                fig, ax = plt.subplots(figsize=figsize)
            except Exception as e:
                print(f"Error creating plot figure: {e}")
                return None, None
            
            # Sort data for smooth line plotting
            try:
                sorted_indices = np.argsort(self.datum_range)
                sorted_datum = self.datum_range[sorted_indices]
                sorted_z0_interval = self.z0_interval[sorted_indices]
            except Exception as e:
                warnings.warn(f"Error sorting data for plotting: {e}")
                sorted_datum = self.datum_range
                sorted_z0_interval = self.z0_interval
            
            # Plot the main curve based on style
            curve_plotted = False
            
            if plot_style == 'smooth' and len(sorted_datum) > 3:
                # Try spline interpolation first
                try:
                    from scipy.interpolate import make_interp_spline
                    spline = make_interp_spline(sorted_datum, sorted_z0_interval, k=min(3, len(sorted_datum)-1))
                    datum_smooth = np.linspace(sorted_datum.min(), sorted_datum.max(), 500)
                    z0_smooth = spline(datum_smooth)
                    
                    # Check for valid interpolation
                    if np.all(np.isfinite(z0_smooth)):
                        ax.plot(datum_smooth, z0_smooth, color='blue', linewidth=2.5, 
                               alpha=0.9, label='Z0 Interval Curve', zorder=3)
                        curve_plotted = True
                    else:
                        raise ValueError("Spline interpolation produced invalid values")
                        
                except Exception as e:
                    if self.verbose:
                        print(f"Spline interpolation failed: {e}, trying linear interpolation")
                    
                    # Fallback to linear interpolation
                    try:
                        from scipy.interpolate import interp1d
                        f_linear = interp1d(sorted_datum, sorted_z0_interval, kind='linear', 
                                          bounds_error=False, fill_value='extrapolate')
                        datum_smooth = np.linspace(sorted_datum.min(), sorted_datum.max(), 300)
                        z0_smooth = f_linear(datum_smooth)
                        
                        # Check for valid interpolation
                        if np.all(np.isfinite(z0_smooth)):
                            ax.plot(datum_smooth, z0_smooth, color='blue', linewidth=2.5, 
                                   alpha=0.9, label='Z0 Interval Curve', zorder=3)
                            curve_plotted = True
                        else:
                            raise ValueError("Linear interpolation produced invalid values")
                            
                    except Exception as e:
                        if self.verbose:
                            print(f"Linear interpolation failed: {e}, using line plot")
            
            # Fallback to line or scatter plot
            if not curve_plotted:
                if plot_style == 'scatter':
                    ax.scatter(sorted_datum, sorted_z0_interval, color='blue', s=60, 
                              alpha=0.8, label='Z0 Interval Points', zorder=3)
                else:  # line plot
                    ax.plot(sorted_datum, sorted_z0_interval, 'b-', linewidth=2, 
                           alpha=0.8, label='Z0 Interval', zorder=3, marker='o', markersize=4)
            
            # Add individual data points if requested
            if show_data_points and plot_style != 'scatter':
                try:
                    ax.scatter(sorted_datum, sorted_z0_interval, marker='x', s=40, 
                              color='gray', alpha=0.8, linewidth=1.5, zorder=4, 
                              label='Data Points')
                except Exception as e:
                    warnings.warn(f"Error plotting data points: {e}")
            
            # Store legend handles and labels in logical order
            legend_handles = []
            legend_labels = []
            
            # Add critical lines and points with error handling - in logical order
            try:
                # 1. ZL (leftmost vertical line)
                zl_line = ax.axvline(x=self.zl, color='red', linestyle='--', linewidth=2, 
                                   alpha=0.8, zorder=2)
                legend_handles.append(zl_line)
                legend_labels.append(f'ZL = {self.zl:.4f}')
                
                # 2. Z0L (horizontal line for minimum Z0)
                z0l_line = ax.axhline(y=self.z0l, color='purple', linestyle=':', linewidth=2, 
                                    alpha=0.7, zorder=1)
                legend_handles.append(z0l_line)
                legend_labels.append(f'Z0L = {self.z0l:.4f}')
                
                # 3. Z0 (central vertical line)
                z0_line = ax.axvline(x=self.z0, color='green', linestyle='-', linewidth=2, 
                                   alpha=0.8, zorder=2)
                legend_handles.append(z0_line)
                legend_labels.append(f'Z0 = {self.z0:.4f}')
                
                # 4. Z0U (horizontal line for maximum Z0)
                z0u_line = ax.axhline(y=self.z0u, color='brown', linestyle=':', linewidth=2, 
                                    alpha=0.7, zorder=1)
                legend_handles.append(z0u_line)
                legend_labels.append(f'Z0U = {self.z0u:.4f}')
                
                # 5. ZU (rightmost vertical line)
                zu_line = ax.axvline(x=self.zu, color='orange', linestyle='--', linewidth=2, 
                                   alpha=0.8, zorder=2)
                legend_handles.append(zu_line)
                legend_labels.append(f'ZU = {self.zu:.4f}')
                
                # Add critical points in logical order
                zl_point = ax.scatter([self.zl], [self.z0l], marker='o', s=150, color='red', 
                                    edgecolor='black', linewidth=2, zorder=5)
                legend_handles.append(zl_point)
                legend_labels.append('(ZL, Z0L)')
                
                z0_point = ax.scatter([self.z0], [self.z0], marker='s', s=150, color='green', 
                                    edgecolor='black', linewidth=2, zorder=5)
                legend_handles.append(z0_point)
                legend_labels.append('(Z0, Z0)')
                
                zu_point = ax.scatter([self.zu], [self.z0u], marker='o', s=150, color='orange', 
                                    edgecolor='black', linewidth=2, zorder=5)
                legend_handles.append(zu_point)
                legend_labels.append('(ZU, Z0U)')
                          
            except Exception as e:
                warnings.warn(f"Error plotting critical lines and points: {e}")
            
            # Set axis limits with robust padding
            try:
                # Use ZL and ZU for x-axis limits with padding
                if hasattr(self, 'zl') and hasattr(self, 'zu'):
                    x_range = self.zu - self.zl
                    if x_range == 0:  # Ensure minimum padding
                        x_range = 0.1  # Minimum range for padding
                    x_padding = x_range * 0.1  # 10% padding or minimum 0.1
                    ax.set_xlim(self.zl - x_padding, self.zu + x_padding)
                
                # Use Z0L and Z0U for y-axis limits with padding
                if hasattr(self, 'z0l') and hasattr(self, 'z0u'):
                    y_range = self.z0u - self.z0l
                    if y_range == 0:  # Ensure minimum padding
                        y_range = 0.1
                    y_padding = y_range * 0.1 # 10% padding or minimum
                    ax.set_ylim(self.z0l - y_padding, self.z0u + y_padding)
                    
            except Exception as e:
                warnings.warn(f"Error setting axis limits: {e}")
            
            # Set labels and title
            try:
                ax.set_xlabel('Datum Value', fontsize=10, fontweight='bold')
                ax.set_ylabel('Z0 Value', fontsize=10, fontweight='bold')
                ax.set_title('EGDF Interval Analysis\n' + 
                            f'Points analyzed: {len(self.z0_interval)} | Style: {plot_style}', 
                            fontsize=10)
                
                # Add grid if requested
                if show_grid:
                    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                
                # Add legend with logical order
                try:
                    # Combine all legend handles and labels
                    ax.legend(legend_handles, legend_labels, bbox_to_anchor=(1.05, 1), 
                             loc='upper left', fontsize=10, title='Interval Points (ZL → ZU)')
                except Exception as e:
                    try:
                        ax.legend(legend_handles, legend_labels, fontsize=10)  # Fallback to default legend
                    except Exception as e2:
                        warnings.warn(f"Could not create legend: {e2}")
                    
            except Exception as e:
                warnings.warn(f"Error setting plot labels and formatting: {e}")
            
            # Adjust layout
            try:
                plt.tight_layout()
            except Exception as e:
                warnings.warn(f"Error adjusting layout: {e}")
            
            # Show plot
            try:
                plt.show()
            except Exception as e:
                warnings.warn(f"Error displaying plot: {e}")
            
            return fig, ax
            
        except Exception as e:
            error_msg = f"Error in plotting interval analysis: {e}"
            if self.verbose:
                print(error_msg)
            warnings.warn(error_msg)
            return None, None

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
            'ZL': float(np.round(self.zl, decimals)),
            'Z0L': float(np.round(self.z0l, decimals)),
            'Z0': float(np.round(self.z0, decimals)),
            'Z0U': float(np.round(self.z0u, decimals)),
            'ZU': float(np.round(self.zu, decimals))
        }
    
    def _fit_egdf_intv(self):
        # try:
        if self.verbose:
            print("\n\nFitting EGDF Interval Analysis...")
            
        # get initial EGDF
        self._get_initial_egdf()

        # get Z0 of the base sample
        self.z0 = self._get_z0(self.init_egdf)

        # compute interval values
        self._compute_intv()

        if self.verbose:
            print("EGDF Interval Analysis fitted successfully.")

        # optional bounds
        
        # except Exception as e:
        #     if self.verbose:
        #         print(f"Error occurred during fitting: {e}")
