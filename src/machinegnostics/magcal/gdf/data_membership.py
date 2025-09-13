'''
DataMembership

- Membership test: "Is a value Zξ a potential member of the given sample Z?" In other words: "Will the homogeneous sample Z remain homogeneous after extension by Zξ"?
- This only works with EGDF
- logic process:
  1. Check if the sample Z is homogeneous using DataHomogeneity. For that first look into egdf.params['is_homogeneous']. If not present, run DataHomogeneity on Z.
  2. If Z is homogeneous, extend egdf.data sample with Zξ in range of [lb, ub] and check if the extended sample remains homogeneous using DataHomogeneity.
  3. We need to find two bounds, lower sample bound LSB and upper sample bound USB. for LSB search range is [LB, DLB] and for USB search range is [DUB, UB]. where DL is the data limit (min and max of Z). LB and UB are the lower and upper bounds of the data universe. 
  4. need to find minimum and maximum values of Zξ that keeps the extended sample homogeneous.

'''
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional
from machinegnostics.magcal.gdf.egdf import EGDF
from machinegnostics.magcal.gdf.homogeneity import DataHomogeneity


class DataMembership:
    
    def __init__(self, 
                 egdf: EGDF,
                 verbose: bool = True,
                 catch: bool = True,
                 tolerance: float = 1e-3,
                 max_iterations: int = 100,
                 initial_step_factor: float = 0.001):
        
        self.egdf = egdf
        self.verbose = verbose
        self.catch = catch
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.initial_step_factor = initial_step_factor
        
        self._validate_egdf()
        
        self.LSB = None
        self.USB = None
        self.is_homogeneous = None
        self._fitted = False
        self.params = {}
        
        if self.catch:
            self.params['errors'] = []
            self.params['warnings'] = []
    
    def _validate_egdf(self):
        if not hasattr(self.egdf, '__class__'):
            raise ValueError("DataMembership: Input must be an EGDF object")
        
        class_name = self.egdf.__class__.__name__
        if 'EGDF' not in class_name:
            raise ValueError(f"DataMembership: Only EGDF objects are supported. Got {class_name}")
        
        if not hasattr(self.egdf, '_fitted') or not self.egdf._fitted:
            raise ValueError("DataMembership: EGDF object must be fitted before membership analysis")
        
        if not hasattr(self.egdf, 'data') or self.egdf.data is None:
            raise ValueError("DataMembership: EGDF object must contain data")
    
    def _append_error(self, error_message: str, exception_type: str = None):
        if self.catch:
            error_entry = {
                'method': 'DataMembership',
                'error': error_message,
                'exception_type': exception_type or 'DataMembershipError'
            }
            self.params['errors'].append(error_entry)
    
    def _append_warning(self, warning_message: str):
        if self.catch:
            warning_entry = {
                'method': 'DataMembership',
                'warning': warning_message
            }
            self.params['warnings'].append(warning_entry)
    
    def _check_original_homogeneity(self) -> bool:
        if self.verbose:
            print("DataMembership: Checking original sample homogeneity...")
        
        if (hasattr(self.egdf, 'params') and 
            self.egdf.params and 
            'is_homogeneous' in self.egdf.params):
            
            is_homogeneous = self.egdf.params['is_homogeneous']
            if self.verbose:
                print(f"DataMembership: Found existing homogeneity result: {is_homogeneous}")
            return is_homogeneous
        
        try:
            if self.verbose:
                print("DataMembership: Running DataHomogeneity analysis...")
            
            homogeneity = DataHomogeneity(
                gdf=self.egdf,
                verbose=self.verbose,
                catch=self.catch
            )
            is_homogeneous = homogeneity.fit()
            
            if self.verbose:
                print(f"DataMembership: Homogeneity analysis result: {is_homogeneous}")
            
            return is_homogeneous
            
        except Exception as e:
            error_msg = f"Error in homogeneity check: {str(e)}"
            self._append_error(error_msg, type(e).__name__)
            if self.verbose:
                print(f"DataMembership: Error: {error_msg}")
            raise
    
    def _test_membership_at_point(self, test_point: float) -> bool:
        try:
            extended_data = np.append(self.egdf.data, test_point)
            
            extended_egdf = EGDF(data=extended_data,
                                 S=self.egdf.S,
                                 verbose=False,
                                 catch=True,
                                 flush=True,
                                 z0_optimize=self.egdf.z0_optimize,
                                 tolerance=self.egdf.tolerance,
                                 data_form=self.egdf.data_form,
                                 n_points=self.egdf.n_points,
                                 homogeneous=self.egdf.homogeneous,
                                 opt_method=self.egdf.opt_method,
                                 max_data_size=self.egdf.max_data_size,
                                 wedf=self.egdf.wedf,
                                 weights=None)
            extended_egdf.fit(plot=False)
            
            homogeneity = DataHomogeneity(
                gdf=extended_egdf,
                verbose=False,
                catch=True
            )
            is_homogeneous = homogeneity.fit()
            
            return is_homogeneous
            
        except Exception as e:
            if self.verbose:
                print(f"DataMembership: Error testing point {test_point:.6f}: {str(e)}")
            return False
    
    def _calculate_adaptive_step(self, data_range: float, iteration: int) -> float:
        base_step = data_range * self.initial_step_factor
        decay_factor = 1.0 / (1.0 + 0.1 * iteration)
        return base_step * decay_factor
    
    def _find_sample_bound(self, bound_type: str) -> Optional[float]:
        if bound_type not in ['lower', 'upper']:
            raise ValueError("bound_type must be either 'lower' or 'upper'")
        
        data_range = self.egdf.DUB - self.egdf.DLB
        
        if bound_type == 'lower':
            search_start = self.egdf.DLB
            search_end = self.egdf.LB if self.egdf.LB is not None else self.egdf.DLB - data_range
            direction = "LSB"
            move_direction = -1
        else:
            search_start = self.egdf.DUB
            search_end = self.egdf.UB if self.egdf.UB is not None else self.egdf.DUB + data_range
            direction = "USB"
            move_direction = 1
        
        if self.verbose:
            print(f"DataMembership: Searching for {direction} from {search_start:.6f} towards {search_end:.6f}")
        
        # Check if the starting point (data boundary) is homogeneous
        first_test = self._test_membership_at_point(search_start)
        
        if not first_test:
            # If data boundary itself is not homogeneous, return the data boundary
            if self.verbose:
                print(f"DataMembership: Data boundary {search_start:.6f} is not homogeneous")
                print(f"DataMembership: {direction} = {search_start:.6f} (data boundary)")
            return search_start
        
        current_point = search_start
        best_bound = search_start
        step_size = self._calculate_adaptive_step(data_range, 0)
        
        for iteration in range(self.max_iterations):
            current_point += move_direction * step_size
            
            # Check bounds
            if bound_type == 'lower' and current_point <= search_end:
                break
            if bound_type == 'upper' and current_point >= search_end:
                break
            
            is_homogeneous = self._test_membership_at_point(current_point)
            
            if self.verbose and iteration % 10 == 0:
                print(f"DataMembership: {direction} iteration {iteration}: "
                      f"testing point {current_point:.6f} (homogeneous: {is_homogeneous})")
            
            if is_homogeneous:
                best_bound = current_point
                # Adaptive step size
                step_size = self._calculate_adaptive_step(data_range, iteration)
            else:
                # Found the boundary where homogeneity is lost
                break
        
        if best_bound is not None:
            if self.verbose:
                print(f"DataMembership: Found {direction} = {best_bound:.6f} after {iteration + 1} iterations")
        else:
            warning_msg = f"Could not find {direction} within search range"
            self._append_warning(warning_msg)
            if self.verbose:
                print(f"DataMembership: Warning: {warning_msg}")
        
        return best_bound
    
    def fit(self) -> Tuple[Optional[float], Optional[float]]:
        try:
            if self.verbose:
                print("DataMembership: Starting membership analysis...")
            
            self.is_homogeneous = self._check_original_homogeneity()
            
            if not self.is_homogeneous:
                error_msg = "Original sample is not homogeneous. Membership analysis requires homogeneous data."
                self._append_error(error_msg)
                if self.verbose:
                    print(f"DataMembership: Error: {error_msg}")
                raise RuntimeError(error_msg)
            
            if self.verbose:
                print("DataMembership: Original sample is homogeneous. Proceeding with bound search...")
            
            if self.verbose:
                print("DataMembership: Finding Lower Sample Bound (LSB)...")
            self.LSB = self._find_sample_bound('lower')
            
            if self.verbose:
                print("DataMembership: Finding Upper Sample Bound (USB)...")
            self.USB = self._find_sample_bound('upper')
            
            if self.catch:
                self.params.update({
                    'LSB': float(self.LSB) if self.LSB is not None else None,
                    'USB': float(self.USB) if self.USB is not None else None,
                    'is_homogeneous': self.is_homogeneous,
                    'membership_fitted': True,
                    'search_parameters': {
                        'tolerance': self.tolerance,
                        'max_iterations': self.max_iterations,
                        'initial_step_factor': self.initial_step_factor
                    }
                })
            
            if hasattr(self.egdf, 'params') and self.egdf.params:
                self.egdf.params.update({
                    'LSB': float(self.LSB) if self.LSB is not None else None,
                    'USB': float(self.USB) if self.USB is not None else None,
                    'membership_checked': True
                })
                
                if self.verbose:
                    print("DataMembership: Results written to EGDF params dictionary")
            
            self._fitted = True
            
            if self.verbose:
                print("DataMembership: Analysis completed successfully")
                if self.LSB is not None:
                    print(f"DataMembership: Lower Sample Bound (LSB) = {self.LSB:.6f}")
                if self.USB is not None:
                    print(f"DataMembership: Upper Sample Bound (USB) = {self.USB:.6f}")
            
            return self.LSB, self.USB
            
        except Exception as e:
            error_msg = f"Error during membership analysis: {str(e)}"
            self._append_error(error_msg, type(e).__name__)
            if self.verbose:
                print(f"DataMembership: Error: {error_msg}")
            raise
    
    def plot(self, 
             plot_smooth: bool = True, 
             plot: str = 'both', 
             bounds: bool = True, 
             figsize: tuple = (12, 8)):
        if not self._fitted:
            raise RuntimeError("DataMembership: Must call fit() before plotting")
        
        if not self.egdf.catch:
            print("DataMembership: Plot is not available with EGDF catch=False")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            # Create a fresh figure
            fig, ax1 = plt.subplots(figsize=figsize)
            
            # Get EGDF data
            x_points = self.egdf.data
            egdf_data = self.egdf.params.get('egdf')
            pdf_data = self.egdf.params.get('pdf')
            
            # Debug info
            if self.verbose:
                print(f"DataMembership: LSB = {self.LSB}, USB = {self.USB}")
                print(f"DataMembership: Data range: {x_points.min():.3f} to {x_points.max():.3f}")
            
            # Plot EGDF if requested
            if plot in ['gdf', 'both'] and egdf_data is not None:
                # Plot EGDF points
                ax1.plot(x_points, egdf_data, 'o', color='blue', label='EGDF', markersize=4)
                
                # Plot smooth EGDF if available
                if (plot_smooth and hasattr(self.egdf, 'di_points_n') and 
                    hasattr(self.egdf, 'egdf_points') and 
                    self.egdf.di_points_n is not None and 
                    self.egdf.egdf_points is not None):
                    ax1.plot(self.egdf.di_points_n, self.egdf.egdf_points, 
                            color='blue', linestyle='-', linewidth=2, alpha=0.8)
                
                ax1.set_ylabel('EGDF', color='blue')
                ax1.tick_params(axis='y', labelcolor='blue')
                ax1.set_ylim(0, 1)
            
            # Plot PDF if requested
            if plot in ['pdf', 'both'] and pdf_data is not None:
                if plot == 'pdf':
                    # PDF only plot
                    ax1.plot(x_points, pdf_data, 'o', color='red', label='PDF', markersize=4)
                    if (plot_smooth and hasattr(self.egdf, 'di_points_n') and 
                        hasattr(self.egdf, 'pdf_points') and
                        self.egdf.di_points_n is not None and 
                        self.egdf.pdf_points is not None):
                        ax1.plot(self.egdf.di_points_n, self.egdf.pdf_points, 
                                color='red', linestyle='-', linewidth=2, alpha=0.8)
                    ax1.set_ylabel('PDF', color='red')
                    ax1.tick_params(axis='y', labelcolor='red')
                    max_pdf = np.max(pdf_data)
                    ax1.set_ylim(0, max_pdf * 1.1)
                else:
                    # Both EGDF and PDF - create second y-axis
                    ax2 = ax1.twinx()
                    ax2.plot(x_points, pdf_data, 'o', color='red', label='PDF', markersize=4)
                    if (plot_smooth and hasattr(self.egdf, 'di_points_n') and 
                        hasattr(self.egdf, 'pdf_points') and
                        self.egdf.di_points_n is not None and 
                        self.egdf.pdf_points is not None):
                        ax2.plot(self.egdf.di_points_n, self.egdf.pdf_points, 
                                color='red', linestyle='-', linewidth=2, alpha=0.8)
                    ax2.set_ylabel('PDF', color='red')
                    ax2.tick_params(axis='y', labelcolor='red')
                    max_pdf = np.max(pdf_data)
                    ax2.set_ylim(0, max_pdf * 1.1)
                    ax2.legend(loc='upper right')
            
            # Add LSB vertical line
            if self.LSB is not None:
                ax1.axvline(x=self.LSB, color='red', linestyle='--', linewidth=1.5, 
                           alpha=0.9, label=f'LSB = {self.LSB:.3f}', zorder=10)
                if self.verbose:
                    print(f"DataMembership: Added LSB line at {self.LSB}")
            
            # Add USB vertical line  
            if self.USB is not None:
                ax1.axvline(x=self.USB, color='blue', linestyle='--', linewidth=1.5,
                           alpha=0.9, label=f'USB = {self.USB:.3f}', zorder=10)
                if self.verbose:
                    print(f"DataMembership: Added USB line at {self.USB}")
            
            # Add membership range shading if both bounds exist
            if self.LSB is not None and self.USB is not None:
                ax1.axvspan(self.LSB, self.USB, alpha=0.05, color='green', 
                           label='Membership Range', zorder=1)
                if self.verbose:
                    print(f"DataMembership: Added membership range shading")
            
            # Add bounds if requested
            if bounds:
                bound_info = [
                    (self.egdf.params.get('DLB'), 'green', '-', 'DLB'),
                    (self.egdf.params.get('DUB'), 'orange', '-', 'DUB'),
                    (self.egdf.params.get('LB'), 'purple', '--', 'LB'),
                    (self.egdf.params.get('UB'), 'brown', '--', 'UB')
                ]
                
                for bound, color, style, name in bound_info:
                    if bound is not None:
                        ax1.axvline(x=bound, color=color, linestyle=style, linewidth=2, 
                                   alpha=0.8, label=f"{name}={bound:.3f}")
            
            # Add Z0 if available
            if hasattr(self.egdf, 'z0') and self.egdf.z0 is not None:
                ax1.axvline(x=self.egdf.z0, color='magenta', linestyle='-.', linewidth=1, 
                           alpha=0.8, label=f'Z0={self.egdf.z0:.3f}')
            
            # Set formatting
            ax1.set_xlabel('Data Points')
            ax1.grid(True, alpha=0.3)
            
            # Set title
            membership_info = []
            if self.LSB is not None:
                membership_info.append(f"LSB={self.LSB:.3f}")
            if self.USB is not None:
                membership_info.append(f"USB={self.USB:.3f}")
            
            if membership_info:
                title = f"EGDF Membership Analysis: {', '.join(membership_info)}"
            else:
                title = "EGDF Membership Analysis"
            
            ax1.set_title(title, fontsize=12)
            
            # Set x-limits with some padding
            data_range = self.egdf.params['DUB'] - self.egdf.params['DLB']
            padding = data_range * 0.1
            ax1.set_xlim(self.egdf.params['DLB'] - padding, self.egdf.params['DUB'] + padding)
            
            # Add legend
            ax1.legend(loc='upper left', bbox_to_anchor=(0, 1))
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            error_msg = f"Error creating plot: {str(e)}"
            self._append_error(error_msg, type(e).__name__)
            if self.verbose:
                print(f"DataMembership: Error: {error_msg}")
            raise
    
    def results(self) -> Dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("DataMembership: No analysis results available. Call fit() method first")
        
        if not self.catch:
            raise RuntimeError("DataMembership: No results stored. Ensure catch=True during initialization")
        
        return self.params.copy()
    
    @property
    def fitted(self) -> bool:
        return self._fitted