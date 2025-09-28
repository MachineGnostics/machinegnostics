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
import logging
from machinegnostics.magcal.util.logging import get_logger
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional
from machinegnostics.magcal.gdf.egdf import EGDF
from machinegnostics.magcal.gdf.homogeneity import DataHomogeneity


class DataMembership:
    """
    DataMembership

    This class provides functionality to test whether a given value can be considered a member of a homogeneous data sample. It uses the EGDF (Empirical Generalized Distribution Function) framework to determine the homogeneity of the data sample and to calculate the bounds within which new data points can be added without disrupting the homogeneity.

    Attributes:
        egdf (EGDF): An instance of the EGDF class containing the data sample and its parameters.
        verbose (bool): If True, detailed logs are printed during execution.
        catch (bool): If True, errors and warnings are stored in the `params` attribute.
        tolerance (float): The tolerance level for numerical calculations.
        max_iterations (int): The maximum number of iterations for bound search.
        initial_step_factor (float): The initial step size factor for adaptive bound search.
        LSB (float): The calculated Lower Sample Bound (LSB).
        USB (float): The calculated Upper Sample Bound (USB).
        is_homogeneous (bool): Indicates whether the original data sample is homogeneous.
        params (dict): Stores results, errors, warnings, and other parameters.
        _fitted (bool): Indicates whether the membership analysis has been completed.

    Methods:
        fit():
            Performs the membership analysis to determine the LSB and USB.
            Returns a tuple of (LSB, USB).

        plot(plot_smooth=True, plot='both', bounds=True, figsize=(12, 8)):
            Generates a plot of the EGDF and PDF with membership bounds and other relevant information.

        results():
            Returns the analysis results stored in the `params` attribute.

        fitted:
            A property that indicates whether the membership analysis has been completed.

    Usage:
        >>> egdf_instance = EGDF(...)
        >>> membership = DataMembership(egdf_instance)
        >>> membership.fit()
        >>> membership.plot()
        >>> results = membership.results()

    Example:
        >>> from machinegnostics.magcal import EGDF, DataMembership
        >>> egdf_instance = EGDF(data=[1.2, 1.5, 1.7, 1.9], S=2.0)
        >>> egdf_instance.fit()
        >>> membership = DataMembership(egdf_instance, verbose=True)
        >>> lsb, usb = membership.fit()
        >>> print(f"Lower Bound: {lsb}, Upper Bound: {usb}")
        >>> membership.plot()
        >>> results = membership.results()
        >>> print(results)
    """
    
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
        
        # Set up logger
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if verbose else logging.WARNING)
        self.logger.debug(f"{self.__class__.__name__} initialized: ")

        # Validate EGDF object
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
        self.logger.debug("Validating EGDF object for DataMembership analysis")
        if not hasattr(self.egdf, '__class__'):
            self.logger.error("Input must be an EGDF object")
            raise ValueError("Input must be an EGDF object")
        
        class_name = self.egdf.__class__.__name__
        if 'EGDF' not in class_name:
            self.logger.error(f"Only EGDF objects are supported. Got {class_name}")
            raise ValueError(f"Only EGDF objects are supported. Got {class_name}")
        
        if not hasattr(self.egdf, '_fitted') or not self.egdf._fitted:
            self.logger.error("EGDF object must be fitted before membership analysis")
            raise ValueError("EGDF object must be fitted before membership analysis")
        
        if not hasattr(self.egdf, 'data') or self.egdf.data is None:
            self.logger.error("EGDF object must contain data")
            raise ValueError("EGDF object must contain data")
    
    def _append_error(self, error_message: str, exception_type: str = None):
        self.logger.error(error_message)
        if self.catch:
            error_entry = {
                'method': 'DataMembership',
                'error': error_message,
                'exception_type': exception_type or 'DataMembershipError'
            }
            self.params['errors'].append(error_entry)
            
    
    def _append_warning(self, warning_message: str):
        self.logger.warning(warning_message)
        if self.catch:
            warning_entry = {
                'method': 'DataMembership',
                'warning': warning_message
            }
            self.params['warnings'].append(warning_entry)
    
    def _check_original_homogeneity(self) -> bool:
        self.logger.info("Checking original sample homogeneity")
        
        if (hasattr(self.egdf, 'params') and 
            self.egdf.params and 
            'is_homogeneous' in self.egdf.params):
            
            is_homogeneous = self.egdf.params['is_homogeneous']
            self.logger.info(f"Found existing homogeneity result: {is_homogeneous}")
            return is_homogeneous
        
        try:
            self.logger.info("Running DataHomogeneity analysis...")
            homogeneity = DataHomogeneity(
                gdf=self.egdf,
                verbose=self.verbose,
                catch=self.catch
            )
            is_homogeneous = homogeneity.fit()

            self.logger.info(f"Homogeneity analysis result: {is_homogeneous}")

            return is_homogeneous
            
        except Exception as e:
            error_msg = f"Error in homogeneity check: {str(e)}"
            self._append_error(error_msg, type(e).__name__)
            raise
    
    def _test_membership_at_point(self, test_point: float) -> bool:
        self.logger.debug(f"Testing membership at point: {test_point:.6f}")
        try:
            extended_data = np.append(self.egdf.data, test_point)
            
            extended_egdf = EGDF(S=self.egdf.S,
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
            extended_egdf.fit(data=extended_data, plot=False)

            homogeneity = DataHomogeneity(
                gdf=extended_egdf,
                verbose=False,
                catch=True
            )
            is_homogeneous = homogeneity.fit()
            
            return is_homogeneous
            
        except Exception as e:
            self.logger.error(f"Error testing point {test_point:.6f}: {str(e)}")
            return False
    
    def _calculate_adaptive_step(self, data_range: float, iteration: int) -> float:
        self.logger.debug(f"Calculating adaptive step size at iteration {iteration}")
        base_step = data_range * self.initial_step_factor
        decay_factor = 1.0 / (1.0 + 0.1 * iteration)
        return base_step * decay_factor
    
    def _find_sample_bound(self, bound_type: str) -> Optional[float]:
        self.logger.info(f"Finding {bound_type} sample bound")
        if bound_type not in ['lower', 'upper']:
            self.logger.error("Invalid bound_type")
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

        self.logger.info(f"Searching for {direction} from {search_start:.6f} towards {search_end:.6f}")

        # Check if the starting point (data boundary) is homogeneous
        first_test = self._test_membership_at_point(search_start)
        
        if not first_test:
            # If data boundary itself is not homogeneous, return the data boundary
            self.logger.info(f"Data boundary {search_start:.6f} is not homogeneous")
            self.logger.info(f"{direction} = {search_start:.6f} (data boundary)")
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

            if iteration % 10 == 0:
                self.logger.info(f"{direction} iteration {iteration}: "
                                 f"testing point {current_point:.6f} (homogeneous: {is_homogeneous})")

            if is_homogeneous:
                best_bound = current_point
                # Adaptive step size
                step_size = self._calculate_adaptive_step(data_range, iteration)
            else:
                # Found the boundary where homogeneity is lost
                break
        
        if best_bound is not None:
            self.logger.info(f"Found {direction} = {best_bound:.6f} after {iteration + 1} iterations")
        else:
            warning_msg = f"Could not find {direction} within search range"
            self._append_warning(warning_msg)
        
        return best_bound
    
    def fit(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Performs the membership analysis to determine the Lower Sample Bound (LSB) and Upper Sample Bound (USB).

        This method checks the homogeneity of the original data sample and calculates the bounds within which new data points can be added without disrupting the homogeneity.

        Returns:
            Tuple[Optional[float], Optional[float]]: The calculated LSB and USB values. Returns None for a bound if it cannot be determined.

        Raises:
            RuntimeError: If the original data sample is not homogeneous.
            Exception: For any other errors encountered during the analysis.
        """
        self.logger.info("Starting membership analysis...")
        try:
            self.is_homogeneous = self._check_original_homogeneity()
            
            if not self.is_homogeneous:
                error_msg = "Original sample is not homogeneous. Membership analysis requires homogeneous data."
                self._append_error(error_msg)
                raise RuntimeError(error_msg)

            self.logger.info("Original sample is homogeneous. Proceeding with bound search...")

            self.logger.info("Finding Lower Sample Bound (LSB)...")
            self.LSB = self._find_sample_bound('lower')

            self.logger.info("Finding Upper Sample Bound (USB)...")
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
                
                self.logger.info("Results written to EGDF params dictionary")
            
            self._fitted = True

            self.logger.info("Analysis completed successfully")
            if self.LSB is not None:
                self.logger.info(f"Lower Sample Bound (LSB) = {self.LSB:.6f}")
            if self.USB is not None:
                self.logger.info(f"Upper Sample Bound (USB) = {self.USB:.6f}")

            return self.LSB, self.USB
            
        except Exception as e:
            error_msg = f"Error during membership analysis: {str(e)}"
            self._append_error(error_msg, type(e).__name__)
            raise
    
    def plot(self, 
             plot_smooth: bool = True, 
             plot: str = 'both', 
             bounds: bool = True, 
             figsize: tuple = (12, 8)):
        """
        Generates a plot of the EGDF and PDF with membership bounds and other relevant information.

        Parameters:
            plot_smooth (bool): If True, plots a smoothed version of the EGDF and PDF.
            plot (str): Specifies what to plot. Options are 'gdf', 'pdf', or 'both'.
            bounds (bool): If True, includes data bounds (DLB, DUB, LB, UB) in the plot.
            figsize (tuple): The size of the plot figure.

        Raises:
            RuntimeError: If the `fit` method has not been called before plotting.
            Exception: For any errors encountered during plotting.
        """
        self.logger.info("Generating membership plot...")

        if not self._fitted:
            self.logger.error("Must call fit() before plotting")
            raise RuntimeError("Must call fit() before plotting")
        
        if not self.egdf.catch:
            self.logger.warning("Plot is not available with EGDF catch=False")
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
            self.logger.info(f"LSB = {self.LSB}, USB = {self.USB}")
            self.logger.info(f"Data range: {x_points.min():.3f} to {x_points.max():.3f}")

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
                self.logger.info(f"Added LSB line at {self.LSB}")

            # Add USB vertical line
            if self.USB is not None:
                ax1.axvline(x=self.USB, color='blue', linestyle='--', linewidth=1.5,
                           alpha=0.9, label=f'USB = {self.USB:.3f}', zorder=10)
                self.logger.info(f"Added USB line at {self.USB}")

            # Add membership range shading if both bounds exist
            if self.LSB is not None and self.USB is not None:
                ax1.axvspan(self.LSB, self.USB, alpha=0.05, color='green', 
                           label='Membership Range', zorder=1)
                self.logger.info("Added membership range shading")
            
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
            raise
    
    def results(self) -> Dict[str, Any]:
        """
        Returns the analysis results stored in the `params` attribute.

        This method provides the calculated LSB, USB, and other relevant parameters, as well as any errors or warnings encountered during the analysis.

        Returns:
            Dict[str, Any]: A dictionary containing the analysis results, errors, warnings, and other parameters.

        Raises:
            RuntimeError: If the `fit` method has not been called before accessing results.
            RuntimeError: If `catch` is set to False during initialization, as no results are stored.
        """
        self.logger.info("Retrieving analysis results...")
        if not self._fitted:
            raise RuntimeError("No analysis results available. Call fit() method first")
        
        if not self.catch:
            raise RuntimeError("No results stored. Ensure catch=True during initialization")
        
        return self.params.copy()
    
    @property
    def fitted(self) -> bool:
        return self._fitted
    
    def __repr__(self):
        return (f"<DataMembership(fitted={self._fitted}, "
                f"LSB={self.LSB}, USB={self.USB}, "
                f"is_homogeneous={self.is_homogeneous})>")