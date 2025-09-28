"""
Z0 Estimator - Universal class for estimating Z0 point for GDF distributions

Z0 is the point where:
- For EGDF/ELDF: PDF reaches its global maximum
- For QLDF/QGDF: Distribution function equals 0.5 (median/50th percentile)

Author: Nirmal Parmar
Machine Gnostics
"""

import numpy as np
import logging
from machinegnostics.magcal.util.logging import get_logger
from typing import Union, Dict, Any, Optional

class Z0Estimator:
    """
    Universal Z0 estimator for all GDF (Gnostics Distribution Function) types.
    
    This class automatically detects the distribution type and finds the appropriate Z0 point:
    - For EGDF/ELDF: Finds the point where PDF reaches its global maximum
    - For QLDF/QGDF: Finds the point where the distribution function equals 0.5
    
    The estimator uses multiple advanced methods including spline optimization, polynomial fitting,
    refined interpolation, and parabolic interpolation to achieve high accuracy.
    
    Key Features:
        - Automatic distribution type detection (EGDF, ELDF, QLDF, QGDF)
        - Multiple estimation strategies (simple discrete vs advanced optimization)
        - Robust handling of flat regions and edge cases
        - Comprehensive diagnostic information
        - Built-in visualization capabilities
        - Automatic Z0 assignment back to GDF object
        - Estimate Z0 gnostic error properties (Residual Entropy, RRE)
    
    Attributes:
        gdf: The fitted GDF object (EGDF, ELDF, QLDF, or QGDF)
        gdf_type (str): Detected distribution type ('egdf', 'eldf', 'qldf', 'qgdf')
        optimize (bool): Whether to use advanced optimization methods
        verbose (bool): Whether to print detailed progress information
        find_median (bool): True for QLDF/QGDF (find 0.5 point), False for EGDF/ELDF (find PDF max)
        z0 (float): Estimated Z0 value (None until fit() is called)
        estimation_info (dict): Detailed information about the estimation process
    
    Usage Patterns:
        
        1. Basic Usage (E-distributions - finds PDF maximum):
        ```python
        from machinegnostics.magcal import EGDF
        from machinegnostics.magcal import Z0Estimator
        
        # Fit your distribution
        egdf = EGDF(data=your_data)
        egdf.fit()
        
        # Estimate Z0
        estimator = Z0Estimator(egdf, verbose=True)
        z0 = estimator.fit()
        print(f"Z0 at PDF maximum: {z0}")
        ```
        
        2. Q-distributions Usage:
        ```python
        from machinegnostics.magcal import QLDF
        from machinegnostics.magcal import Z0Estimator
        
        # Fit your Q-distribution
        qldf = QLDF(data=your_data)
        qldf.fit()
        
        # Estimate Z0 at median (0.5)
        estimator = Z0Estimator(qldf, optimize=True, verbose=True)
        z0 = estimator.fit()
        print(f"Z0 at median (0.5): {z0}")
        ```
        
        3. Simple vs Advanced Estimation:
        ```python
        # Fast discrete estimation (good for quick analysis)
        estimator_simple = Z0Estimator(gdf_object, optimize=False)
        z0_simple = estimator_simple.fit()
        
        # Advanced optimization (higher accuracy, slower)
        estimator_advanced = Z0Estimator(gdf_object, optimize=True, verbose=True)
        z0_advanced = estimator_advanced.fit()
        ```
        
        4. Getting Detailed Information:
        ```python
        # Get estimation details
        info = estimator.get_estimation_info()
        print(f"Method used: {info['z0_method']}")
        print(f"Target type: {info['target_type']}")
        print(f"Distribution type: {info['gdf_type']}")
        
        # Check what the estimator is looking for
        if estimator.find_median:
            print("Looking for median (0.5 point)")
        else:
            print("Looking for PDF maximum")
        ```
        
        5. Visualization:
        ```python
        # Create diagnostic plots
        estimator.plot_z0_analysis()
        # Shows PDF with Z0 point + distribution function/CDF
        ```
    
    Advanced Methods (when optimize=True):
        
    For Q-distributions (median finding):
        - Spline interpolation with root finding
        - Linear interpolation between bracketing points
        - Polynomial fitting with root solving
        
    For E-distributions (PDF maximum finding):
        - Spline optimization over entire domain
        - Polynomial fitting with critical point analysis
        - Refined interpolation with fine grid search
        - Parabolic interpolation using three-point method
    
    Error Handling:
        - Validates GDF object is properly fitted
        - Checks for required data based on distribution type
        - Graceful fallback to discrete methods if advanced methods fail
        - Clear error messages for common issues
    
    Performance Notes:
        - Simple mode (optimize=False): Very fast, good accuracy for most cases
        - Advanced mode (optimize=True): Higher accuracy, ~2-10x slower depending on data size
        - Memory usage scales linearly with number of evaluation points
        - Recommended to use verbose=True for diagnostic purposes
    
    Notes:
        - The GDF object must be fitted before passing to Z0Estimator
        - For Q-distributions: finds where distribution function = 0.5 (median/50th percentile)
        - For E-distributions: finds where PDF reaches its global maximum
        - Advanced methods are tried in order of sophistication and reliability
        - The estimated Z0 is automatically assigned back to the GDF object
        - All methods handle flat regions by finding the middle point
        - Works with any GDF subclass that follows the standard interface
    
    Examples:
        
        Complete workflow example:
        ```python
        import numpy as np
        from machinegnostics.magcal import EGDF
        from machinegnostics.magcal import Z0Estimator
        
        # Generate some sample data
        data = np.random.normal(0, 1, 1000)
        
        # Fit EGDF
        egdf = EGDF(data=data)
        egdf.fit()
        
        # Estimate Z0 with detailed output
        estimator = Z0Estimator(egdf, optimize=True, verbose=True)
        z0 = estimator.fit()
        
        # Check results
        print(f"\\nEstimated Z0: {z0:.6f}")
        print(f"Original GDF Z0: {egdf.z0:.6f}")  # Automatically updated
        
        # Get detailed info
        info = estimator.get_estimation_info()
        print(f"Method used: {info['z0_method']}")
        
        # Visualize results
        estimator.plot_z0_analysis()
        ```
    """
    
    def __init__(self,
                 gdf_object: object,
                 optimize: bool = True,
                 verbose: bool = False):
        """
        Initialize the Z0 estimator.
        
        Args:
            gdf_object: A fitted GDF object (EGDF, ELDF, QLDF, or QGDF)
                       Must have been fitted (gdf_object.fit() called) before passing here.
            optimize (bool, optional): Whether to use advanced optimization methods.
                                     If True, uses spline optimization, polynomial fitting, etc.
                                     If False, uses simple discrete search.
                                     Defaults to True.
            verbose (bool, optional): Whether to print detailed progress information
                                    during the estimation process. Defaults to False.
        
        Raises:
            ValueError: If gdf_object is not fitted or doesn't contain required data
            
        Examples:
            >>> # With advanced optimization (recommended)
            >>> estimator = Z0Estimator(fitted_gdf, optimize=True, verbose=True)
            
            >>> # Simple discrete estimation (faster)
            >>> estimator = Z0Estimator(fitted_gdf, optimize=False)
        """
        
        self._validate_gdf_object(gdf_object)
        
        self.gdf = gdf_object
        self.gdf_type = self._detect_gdf_type()
        self.optimize = optimize
        self.verbose = verbose

        # Determine what we're looking for
        self.find_median = self.gdf_type.lower() in ['qldf', 'qgdf']
        
        # Results storage
        self.z0 = None
        self.estimation_info = {}

        # logger
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if verbose else logging.WARNING)
        self.logger.debug(f"{self.__class__.__name__} initialized:")
    
    def _compute_error_properties_for_mean(self, z0):
        """
        Compute error properties at the given Z0 point.
        """
        self.logger.info("Computing error properties at Z0.")
        # estimate q and q1
        gc, q, q1 = self.gdf._calculate_gcq_at_given_zi(z0)

        # fi and fj
        fi_z0 = gc._fi(q, q1) # GME Gnostic Mean Estimating
        fj_z0 = gc._fj(q, q1) # GMQ Gnostic Mean Quantifying

        # entropy
        i_e = np.mean(gc._ientropy(fj_z0))
        j_e = np.mean(gc._jentropy(fj_z0))
        self.residual_entropy = np.mean(gc._rentropy(i_e, j_e))

        # RRE Relative Residual Entropy
        self.RRE = np.mean((fj_z0 - fi_z0) / (fj_z0 + fi_z0))

        # store to given gdf params
        if hasattr(self.gdf, 'params'):
            self.gdf.params['residual_entropy'] = float(self.residual_entropy)
            self.gdf.params['RRE'] = float(self.RRE)
        self.logger.info(f"Computed Residual Entropy: {self.residual_entropy}, RRE: {self.RRE}")

    def fit(self) -> float:
        """
        Estimate the Z0 point.
        
        For EGDF/ELDF distributions, finds the point where PDF reaches its global maximum.
        For QLDF/QGDF distributions, finds the point where the distribution function equals 0.5.
        
        Returns:
            float: The estimated Z0 value
            
        Raises:
            ValueError: If required data is not available for estimation
            
        Examples:
            >>> z0 = estimator.fit()
            >>> print(f"Estimated Z0: {z0:.6f}")
            
            >>> # The Z0 is automatically assigned to the GDF object
            >>> print(f"GDF Z0: {estimator.gdf.z0:.6f}")
        
        Notes:
            - For Q-distributions: finds closest point to 0.5 in distribution function
            - For E-distributions: finds PDF maximum (existing logic)
            - Advanced methods adapt to the target type automatically
            - The estimated Z0 is automatically assigned to the original GDF object
        """
        # Add the safe Z0 estimating trick here
        if np.all(self.gdf.data == self.gdf.data[0]):
            self.logger.info("All data values are the same. Returning the mean value as Z0.")
            self.z0 = np.mean(self.gdf.data)
            self.gdf.z0 = self.z0  # Assign Z0 back to the GDF object
            return self.z0
        
        self.logger.info("Fitting Z0 estimator.")
        if self.find_median:
            self.logger.info(f"Finding Z0 where {self.gdf_type.upper()} = 0.5 (median)")
            self.z0 = self._fit_median()
        else:
            self.logger.info(f"Finding Z0 where {self.gdf_type.upper()} PDF reaches maximum")  
            self.z0 = self._fit_pdf_maximum()
        
        # error in mean
        self.logger.info("Computing error properties at estimated Z0.")
        self._compute_error_properties_for_mean(self.z0)
        self.logger.info(f"Estimated Z0.")
        return self.z0
        

    def _fit_median(self) -> float:
        """Find Z0 where Q-distribution equals 0.5 (median)."""
        self.logger.info(f"Finding Z0 where {self.gdf_type.upper()} = 0.5 (median)")

        # Get distribution function points and data points
        dist_points = self._get_distribution_points()
        di_points = self._get_di_points()
        
        if len(dist_points) == 0:
            self.logger.error("No distribution function data available for Z0 estimation")
            raise ValueError("No distribution function data available for Z0 estimation")
        
        # Find the point closest to 0.5
        target_value = 0.5
        diff_from_target = np.abs(dist_points - target_value)
        closest_idx = np.argmin(diff_from_target)
        closest_value = dist_points[closest_idx]
        closest_location = di_points[closest_idx]

        self.logger.info(f"Discrete closest to 0.5: {self.gdf_type.upper()}={closest_value:.6f} at x={closest_location:.6f} (index {closest_idx})")

        if self.optimize:
            self.z0 = self._find_z0_advanced_median(closest_idx, di_points, dist_points, target_value)
            
            method_used = self._get_last_method_used()
            self.logger.info(f"Advanced estimation complete. Method: {method_used}")
        else:
            self.z0 = closest_location
            
            # Store simple estimation info
            self.estimation_info = {
                'z0': self.z0,
                'z0_method': 'discrete_closest_to_median',
                'z0_target_value': target_value,
                'z0_actual_value': closest_value,
                'z0_target_index': closest_idx,
                'gdf_type': self.gdf_type,
                'target_type': 'median (0.5)'
            }

            self.logger.info(f"Simple estimation: Using discrete closest to 0.5")

        # Update GDF object with Z0
        self.gdf.z0 = self.z0
        if hasattr(self.gdf, 'catch') and self.gdf.catch and hasattr(self.gdf, 'params'):
            self.gdf.params['z0'] = float(self.z0)
        
        return self.z0
    
    def _fit_pdf_maximum(self) -> float:
        """Find Z0 where PDF reaches maximum (existing logic for EGDF/ELDF)."""
        self.logger.info(f"Finding Z0 where {self.gdf_type.upper()} PDF reaches maximum")

        # Get PDF and data points
        pdf_points = self._get_pdf_points()
        di_points = self._get_di_points()
        
        if len(pdf_points) == 0:
            self.logger.error("No PDF data available for Z0 estimation")
            raise ValueError("No PDF data available for Z0 estimation")
        
        # Find the global maximum in the discrete data
        global_max_idx = np.argmax(pdf_points)
        # Handle flat top case - find middle of maximum region
        global_max_idx = self._find_middle_of_flat_region(pdf_points, global_max_idx, find_min=False)
        
        global_max_value = pdf_points[global_max_idx]
        global_max_location = di_points[global_max_idx]

        self.logger.info(f"Discrete global maximum: PDF={global_max_value:.6f} at x={global_max_location:.6f} (index {global_max_idx})")

        if self.optimize:
            z0_candidate = self._find_z0_advanced_pdf_max(global_max_idx, di_points, pdf_points)
            # Check if advanced method is close to discrete maximum
            if abs(z0_candidate - global_max_location) > 1e-6:
                self.logger.info(f"Advanced method z0 ({z0_candidate}) differs from discrete max ({global_max_location}), using discrete max.")
                self.z0 = global_max_location
                self.estimation_info['z0_method'] = 'discrete_pdf_maximum'
            else:
                self.z0 = z0_candidate
            
            # Store simple estimation info
            self.estimation_info = {
                'z0': self.z0,
                'z0_method': 'discrete_pdf_maximum',
                'z0_extremum_pdf_value': global_max_value,
                'z0_extremum_pdf_index': global_max_idx,
                'gdf_type': self.gdf_type,
                'target_type': 'pdf_maximum'
            }

            self.logger.info(f"Simple estimation: Using discrete PDF maximum at Z0")

        # Update GDF object with Z0
        self.gdf.z0 = self.z0
        if hasattr(self.gdf, 'catch') and self.gdf.catch and hasattr(self.gdf, 'params'):
            self.gdf.params['z0'] = float(self.z0)
        
        return self.z0
    
    def get_estimation_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the Z0 estimation process.
        
        Returns comprehensive information about how the Z0 value was estimated,
        including the method used, target type, and various diagnostic values.
        
        Returns:
            Dict[str, Any]: Dictionary containing estimation details:
                - z0 (float): The estimated Z0 value
                - z0_method (str): Method used for estimation
                - gdf_type (str): Type of distribution ('egdf', 'eldf', 'qldf', 'qgdf')
                - target_type (str): What was being optimized ('median (0.5)' or 'pdf_maximum')
                - Additional fields depending on the target type
        
        Examples:
            >>> estimator.fit()
            >>> info = estimator.get_estimation_info()
            >>> print(f"Z0: {info['z0']:.6f}")
            >>> print(f"Method: {info['z0_method']}")
            >>> print(f"Target: {info['target_type']}")
        """
        self.logger.info("Retrieving estimation information.")
        if not self.estimation_info:
            return {"error": "No estimation performed yet. Call fit() first."}
        return self.estimation_info.copy()
    
    def plot_z0_analysis(self, figsize: tuple = (12, 6)) -> None:
        """
        Create visualization plots showing the Z0 estimation results.
        
        Generates a two-panel plot showing:
        1. PDF curve with the estimated Z0 point marked
        2. Distribution function curve (for Q-distributions) or CDF (for E-distributions)
        
        Args:
            figsize (tuple, optional): Figure size as (width, height) in inches.
                                     Defaults to (12, 6).
        """
        self.logger.info("Creating Z0 analysis plots.")
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.error("Matplotlib not available. Cannot create plots.")
            return
            
        if self.z0 is None:
            self.logger.error("No Z0 estimation available. Call fit() first.")
            return
        
        pdf_points = self._get_pdf_points()
        di_points = self._get_di_points()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: PDF with Z0 point
        ax1.plot(di_points, pdf_points, 'b-', linewidth=2, label='PDF')
        target_desc = "Median (0.5)" if self.find_median else "PDF Maximum"
        ax1.axvline(self.z0, color='red', linestyle='--', linewidth=2, 
                   label=f'Z0 ({target_desc}): {self.z0:.4f}')
        ax1.scatter([self.z0], [np.interp(self.z0, di_points, pdf_points)], 
                   color='red', s=100, zorder=5)
        ax1.set_xlabel('Value')
        ax1.set_ylabel('PDF')
        ax1.set_title(f'{self.gdf_type.upper()} PDF with Z0 Point')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Distribution function or CDF
        if self.find_median:
            # Plot Q-distribution function
            dist_points = self._get_distribution_points()
            if len(dist_points) > 0:
                ax2.plot(di_points, dist_points, 'g-', linewidth=2, label=f'{self.gdf_type.upper()}')
                ax2.axhline(0.5, color='orange', linestyle=':', linewidth=2, label='Target (0.5)')
                ax2.axvline(self.z0, color='red', linestyle='--', linewidth=2, 
                           label=f'Z0: {self.z0:.4f}')
                ax2.scatter([self.z0], [0.5], color='red', s=100, zorder=5)
                ax2.set_xlabel('Value')
                ax2.set_ylabel(f'{self.gdf_type.upper()}')
                ax2.set_title(f'{self.gdf_type.upper()} with Z0 at Median')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            else:
                self._plot_info_panel(ax2)
        else:
            # Plot CDF if available for E-distributions
            if hasattr(self.gdf, 'cdf_points') and self.gdf.cdf_points is not None:
                ax2.plot(di_points, self.gdf.cdf_points, 'g-', linewidth=2, label='CDF')
                ax2.axvline(self.z0, color='red', linestyle='--', linewidth=2, 
                           label=f'Z0: {self.z0:.4f}')
                ax2.set_xlabel('Value')
                ax2.set_ylabel('CDF')
                ax2.set_title(f'{self.gdf_type.upper()} CDF with Z0 Point')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            else:
                self._plot_info_panel(ax2)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_info_panel(self, ax):
        """Plot estimation information panel."""
        self.logger.info("Creating Z0 estimation information panel.")
        target_desc = "Median (0.5)" if self.find_median else "PDF Maximum"
        info_text = f"Z0 Estimation Info:\n"
        info_text += f"Value: {self.z0:.6f}\n"
        info_text += f"Method: {self.estimation_info.get('z0_method', 'unknown')}\n"
        info_text += f"Target: {target_desc}\n"
        info_text += f"Distribution: {self.gdf_type.upper()}"
        ax.text(0.1, 0.5, info_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_title('Z0 Estimation Information')
    
    def _find_middle_of_flat_region(self, values, extremum_idx, find_min=True):
        """Find the middle point of a flat region (for PDF maximum finding)."""
        self.logger.info("Checking for flat region around extremum index.")

        n_points = len(values)
        extremum_value = values[extremum_idx]
        
        # Define tolerance for "flatness"
        tolerance = np.std(values) * 0.01  # 1% of standard deviation
        tolerance = max(tolerance, 1e-10)  # Minimum tolerance
        
        # Find the range of indices with similar values
        similar_mask = np.abs(values - extremum_value) <= tolerance
        similar_indices = np.where(similar_mask)[0]
        
        if len(similar_indices) > 1:
            # Find continuous regions
            diff_indices = np.diff(similar_indices)
            break_points = np.where(diff_indices > 1)[0]
            
            if len(break_points) == 0:
                # Single continuous region
                middle_idx = similar_indices[len(similar_indices) // 2]
                if self.verbose:
                    region_type = "minimum" if find_min else "maximum"
                    self.logger.info(f"Flat {region_type} region detected. Using middle point at index {middle_idx}")
                return middle_idx
            else:
                # Multiple regions - find the one containing original extremum_idx
                start_idx = 0
                for break_point in break_points:
                    region_indices = similar_indices[start_idx:break_point + 1]
                    if extremum_idx in region_indices:
                        middle_idx = region_indices[len(region_indices) // 2]
                        return middle_idx
                    start_idx = break_point + 1
                
                # Check last region
                region_indices = similar_indices[start_idx:]
                if extremum_idx in region_indices:
                    middle_idx = region_indices[len(region_indices) // 2]
                    return middle_idx
        
        # If no flat region found or single point, return original index
        return extremum_idx
    
    def _validate_gdf_object(self, gdf_object):
        if not hasattr(gdf_object, '_fitted'):
            self.logger.error("GDF object must have '_fitted' attribute")
            raise ValueError("GDF object must have '_fitted' attribute")
        
        if not gdf_object._fitted:
            self.logger.error("GDF object must be fitted before Z0 estimation")
            raise ValueError("GDF object must be fitted before Z0 estimation")
        
        # Check for required data based on distribution type
        temp_gdf_type = self._detect_gdf_type_from_object(gdf_object)
        
        if temp_gdf_type.lower() in ['qldf', 'qgdf']:
            # For Q-distributions, need distribution function data
            has_dist_data = (hasattr(gdf_object, 'cdf_points') and gdf_object.cdf_points is not None) or \
                           (hasattr(gdf_object, 'qgdf_points') and gdf_object.qgdf_points is not None) or \
                           (hasattr(gdf_object, 'qldf_points') and gdf_object.qldf_points is not None)
            if not has_dist_data:
                self.logger.error("Q-distribution object must contain distribution function data")
                raise ValueError("Q-distribution object must contain distribution function data")
        else:
            # For E-distributions, need PDF data
            has_pdf_points = hasattr(gdf_object, 'pdf_points') and gdf_object.pdf_points is not None
            has_pdf = hasattr(gdf_object, 'pdf') and gdf_object.pdf is not None
            if not (has_pdf_points or has_pdf):
                self.logger.error("E-distribution object must contain PDF data")
                raise ValueError("E-distribution object must contain PDF data")
        
        # Check for data points
        has_di_points = hasattr(gdf_object, 'di_points_n') and gdf_object.di_points_n is not None
        has_data = hasattr(gdf_object, 'data') and gdf_object.data is not None
        
        if not (has_di_points or has_data):
            self.logger.error("GDF object must contain data points (di_points_n or data attribute)")
            raise ValueError("GDF object must contain data points (di_points_n or data attribute)")
    
    def _detect_gdf_type(self):
        return self._detect_gdf_type_from_object(self.gdf)
    
    def _detect_gdf_type_from_object(self, gdf_object):
        """Detect the type of GDF distribution from the object class name."""
        class_name = gdf_object.__class__.__name__.lower()
        
        if 'egdf' in class_name:
            return 'egdf'
        elif 'eldf' in class_name:
            return 'eldf'
        elif 'qgdf' in class_name:
            return 'qgdf'
        elif 'qldf' in class_name:
            return 'qldf'
        else:
            # Fallback - assume E-distribution for unknown types
            return 'unknown'
    
    def _find_z0_advanced_median(self, closest_idx, di_points, dist_points, target_value):
        """Find Z0 using advanced methods to locate where distribution = 0.5."""
        
        # Store basic info for all methods
        closest_dist_value = dist_points[closest_idx]
        closest_location = di_points[closest_idx]
        
        self.estimation_info = {
            'z0': None,  # Will be updated
            'z0_method': 'discrete_closest_to_median',  # Will be updated if advanced method succeeds
            'z0_target_value': target_value,
            'z0_actual_value': closest_dist_value,
            'gdf_type': self.gdf_type,
            'target_type': 'median (0.5)',
            'closest_idx': closest_idx,
            'closest_location': closest_location,
            'z0_interpolation_points': len(di_points)
        }
        
        # Try advanced methods in order of preference
        advanced_methods = [
            self._try_spline_median_finding,
            self._try_linear_interpolation_median,
            self._try_polynomial_median_finding
        ]
        
        for method in advanced_methods:
            try:
                result = method(di_points, dist_points, target_value)
                if result is not None:
                    self.estimation_info['z0'] = result
                    return result
            except Exception as e:
                if self.verbose:
                    self.logger.error(f"Method {method.__name__} failed: {e}")
                continue
        
        # All advanced methods failed - use discrete closest
        self.logger.info("All advanced methods failed. Using discrete closest to 0.5.")
        
        self.estimation_info['z0'] = closest_location
        return closest_location
    
    def _find_z0_advanced_pdf_max(self, global_max_idx, di_points, pdf_points):
        """Find Z0 using advanced methods for PDF maximum (existing logic)."""
        
        # Store basic info for all methods
        max_value = pdf_points[global_max_idx]
        max_location = di_points[global_max_idx]
        
        self.estimation_info = {
            'z0': None,  # Will be updated
            'z0_method': 'discrete_pdf_maximum',  # Will be updated if advanced method succeeds
            'z0_extremum_pdf_value': max_value,
            'gdf_type': self.gdf_type,
            'target_type': 'pdf_maximum',
            'global_extremum_idx': global_max_idx,
            'global_extremum_location': max_location,
            'z0_interpolation_points': len(di_points)
        }
        
        # Try advanced methods in order of preference (existing logic)
        advanced_methods = [
            self._try_spline_optimization_pdf,
            self._try_polynomial_fitting_pdf,
            self._try_refined_interpolation_pdf,
            self._try_parabolic_interpolation_pdf
        ]
        
        for method in advanced_methods:
            try:
                result = method(di_points, pdf_points, global_max_idx)
                if result is not None:
                    self.estimation_info['z0'] = result
                    return result
            except Exception as e:
                if self.verbose:
                    self.logger.error(f"Method {method.__name__} failed: {e}")
                continue
        
        # All advanced methods failed - use discrete maximum
        if self.verbose:
            self.logger.info("All advanced methods failed. Using discrete PDF maximum.")
        
        self.estimation_info['z0'] = max_location
        return max_location
    
    def _try_spline_median_finding(self, di_points, dist_points, target_value):
        """Use spline interpolation to find where distribution = target_value."""
        try:
            from scipy.interpolate import UnivariateSpline
            from scipy.optimize import brentq
        except ImportError:
            if self.verbose:
                self.logger.error("SciPy not available for spline median finding")
            return None
        
        try:
            # Create spline interpolation
            spline = UnivariateSpline(di_points, dist_points, s=0, k=3)
            
            # Define function to find root of (spline(x) - target_value)
            def target_function(x):
                return spline(x) - target_value
            
            # Find domain where we cross the target value
            domain_min, domain_max = np.min(di_points), np.max(di_points)
            
            # Check if target value is within the range
            spline_min, spline_max = np.min(dist_points), np.max(dist_points)
            if not (spline_min <= target_value <= spline_max):
                return None
            
            # Use root finding to locate exact crossing
            try:
                z0_candidate = brentq(target_function, domain_min, domain_max)
                
                if domain_min <= z0_candidate <= domain_max:
                    self.estimation_info['z0_method'] = 'spline_median_finding'
                    if self.verbose:
                        self.logger.info(f"Spline median finding successful: Z0={z0_candidate:.8f} (target={target_value})")
                    return z0_candidate
            except ValueError:
                # Try linear search if brentq fails
                fine_x = np.linspace(domain_min, domain_max, 10000)
                fine_y = spline(fine_x)
                closest_idx = np.argmin(np.abs(fine_y - target_value))
                z0_candidate = fine_x[closest_idx]
                
                self.estimation_info['z0_method'] = 'spline_median_search'
                if self.verbose:
                    self.logger.info(f"Spline median search successful: Z0={z0_candidate:.8f} (target={target_value})")
                return z0_candidate
            
        except Exception as e:
            if self.verbose:
                self.logger.error(f"Spline median finding failed: {e}")
        
        return None
    
    def _try_linear_interpolation_median(self, di_points, dist_points, target_value):
        """Use linear interpolation to find where distribution = target_value."""
        try:
            # Find the interval containing the target value
            for i in range(len(dist_points) - 1):
                y1, y2 = dist_points[i], dist_points[i + 1]
                
                # Check if target is between these two points
                if (y1 <= target_value <= y2) or (y2 <= target_value <= y1):
                    x1, x2 = di_points[i], di_points[i + 1]
                    
                    # Linear interpolation
                    if abs(y2 - y1) < 1e-15:  # Avoid division by zero
                        z0_candidate = (x1 + x2) / 2  # Take midpoint if flat
                    else:
                        # Linear interpolation formula
                        t = (target_value - y1) / (y2 - y1)
                        z0_candidate = x1 + t * (x2 - x1)
                    
                    self.estimation_info['z0_method'] = 'linear_interpolation_median'
                    if self.verbose:
                        self.logger.info(f"Linear interpolation median successful: Z0={z0_candidate:.8f} (target={target_value})")
                    return z0_candidate
            
        except Exception as e:
            if self.verbose:
                self.logger.error(f"Linear interpolation median failed: {e}")
        
        return None
    
    def _try_polynomial_median_finding(self, di_points, dist_points, target_value):
        """Use polynomial fitting to find where distribution = target_value."""
        try:
            # Try different polynomial degrees
            for degree in [3, 2]:
                if len(di_points) > degree + 1:
                    try:
                        coeffs = np.polyfit(di_points, dist_points, degree)
                        poly = np.poly1d(coeffs)
                        
                        # Create target function
                        target_poly = poly - target_value
                        roots = np.roots(target_poly)
                        
                        # Filter real roots within domain
                        real_roots = roots[np.isreal(roots)].real
                        domain_min, domain_max = np.min(di_points), np.max(di_points)
                        valid_roots = real_roots[(real_roots >= domain_min) & (real_roots <= domain_max)]
                        
                        if len(valid_roots) > 0:
                            # Choose the root closest to the discrete solution
                            closest_idx = np.argmin(np.abs(dist_points - target_value))
                            discrete_location = di_points[closest_idx]
                            
                            root_distances = np.abs(valid_roots - discrete_location)
                            best_root_idx = np.argmin(root_distances)
                            z0_candidate = valid_roots[best_root_idx]
                            
                            self.estimation_info['z0_method'] = f'polynomial_median_degree_{degree}'
                            if self.verbose:
                                self.logger.info(f"Polynomial median finding (degree {degree}) successful: Z0={z0_candidate:.8f} (target={target_value})")
                            return z0_candidate
                    
                    except (np.linalg.LinAlgError, ValueError):
                        continue
            
        except Exception as e:
            if self.verbose:
                self.logger.error(f"Polynomial median finding failed: {e}")
        
        return None
    
    # Keep existing PDF optimization methods for E-distributions
    def _try_spline_optimization_pdf(self, di_points, pdf_points, global_extremum_idx):
        try:
            from scipy.interpolate import UnivariateSpline
            from scipy.optimize import minimize_scalar
        except ImportError:
            if self.verbose:
                self.logger.error("SciPy not available for spline optimization")
            return None
        
        try:
            # Create spline interpolation
            spline = UnivariateSpline(di_points, pdf_points, s=0, k=3)
            
            # Define objective function (maximize PDF)
            objective = lambda x: -spline(x)
            
            # Optimize over entire domain
            domain_min, domain_max = np.min(di_points), np.max(di_points)
            result = minimize_scalar(objective, bounds=(domain_min, domain_max), method='bounded')
            
            if result.success:
                z0_candidate = result.x
                
                # Validate result
                if domain_min <= z0_candidate <= domain_max:
                    self.estimation_info['z0_method'] = 'global_spline_optimization'
                    if self.verbose:
                        self.logger.info(f"Spline optimization successful: Z0={z0_candidate:.8f} (PDF maximum)")
                    return z0_candidate
            
        except Exception as e:
            if self.verbose:
                self.logger.error(f"Spline optimization failed: {e}")
        
        return None
    
    def _try_polynomial_fitting_pdf(self, di_points, pdf_points, global_extremum_idx):
        """Try polynomial fitting around the PDF maximum region."""
        n_points = len(di_points)
        
        # Define window around extremum (larger for polynomial fitting)
        window_size = min(max(n_points // 4, 5), n_points)
        start_idx = max(0, global_extremum_idx - window_size // 2)
        end_idx = min(n_points, start_idx + window_size)
        start_idx = max(0, end_idx - window_size)  # Adjust if near end
        
        window_x = di_points[start_idx:end_idx]
        window_y = pdf_points[start_idx:end_idx]
        
        if len(window_x) < 5:
            return None
        
        try:
            # Try different polynomial degrees
            for degree in [4, 3, 2]:
                if len(window_x) > degree + 1:
                    try:
                        coeffs = np.polyfit(window_x, window_y, degree)
                        poly = np.poly1d(coeffs)
                        
                        # Find critical points
                        poly_deriv = np.polyder(poly)
                        critical_points = np.roots(poly_deriv)
                        
                        # Filter real critical points within window
                        real_criticals = critical_points[np.isreal(critical_points)].real
                        valid_criticals = real_criticals[(real_criticals >= window_x[0]) & 
                                                       (real_criticals <= window_x[-1])]
                        
                        if len(valid_criticals) > 0:
                            # Evaluate polynomial at critical points
                            critical_values = poly(valid_criticals)
                            
                            # Find the maximum
                            best_idx = np.argmax(critical_values)
                            z0_candidate = valid_criticals[best_idx]
                            
                            # Validate using second derivative test
                            poly_second_deriv = np.polyder(poly_deriv)
                            second_deriv_value = poly_second_deriv(z0_candidate)
                            
                            # Check if it's a maximum
                            if second_deriv_value < 0:
                                self.estimation_info['z0_method'] = f'global_polynomial_fitting_degree_{degree}'
                                if self.verbose:
                                    self.logger.info(f"Polynomial fitting (degree {degree}) successful: Z0={z0_candidate:.8f} (PDF maximum)")
                                return z0_candidate
                    
                    except (np.linalg.LinAlgError, ValueError):
                        continue
            
        except Exception as e:
            if self.verbose:
                self.logger.error(f"Polynomial fitting failed: {e}")
        
        return None
    
    def _try_refined_interpolation_pdf(self, di_points, pdf_points, global_extremum_idx):
        try:
            from scipy.interpolate import interp1d
        except ImportError:
            return None
        
        n_points = len(di_points)
        
        # Define window around extremum
        window_size = min(max(n_points // 6, 3), n_points)
        start_idx = max(0, global_extremum_idx - window_size // 2)
        end_idx = min(n_points, start_idx + window_size)
        start_idx = max(0, end_idx - window_size)
        
        window_x = di_points[start_idx:end_idx]
        window_y = pdf_points[start_idx:end_idx]
        
        if len(window_x) < 4:
            return None
        
        try:
            # Create high-resolution interpolation
            interp_func = interp1d(window_x, window_y, kind='cubic')
            
            # Create fine grid
            fine_x = np.linspace(window_x[0], window_x[-1], len(window_x) * 50)
            fine_y = interp_func(fine_x)
            
            # Find maximum in fine grid
            fine_max_idx = np.argmax(fine_y)
            z0_candidate = fine_x[fine_max_idx]
            
            self.estimation_info['z0_method'] = 'global_refined_interpolation'
            if self.verbose:
                self.logger.info(f"Refined interpolation successful: Z0={z0_candidate:.8f} (PDF maximum)")
            return z0_candidate
            
        except Exception as e:
            if self.verbose:
                self.logger.error(f"Refined interpolation failed: {e}")
        
        return None
    
    def _try_parabolic_interpolation_pdf(self, di_points, pdf_points, global_extremum_idx):
        n_points = len(di_points)
        
        if global_extremum_idx == 0 or global_extremum_idx == n_points - 1:
            return None  # Cannot do parabolic interpolation at boundaries
        
        # Use three points around extremum
        x1, x2, x3 = di_points[global_extremum_idx-1:global_extremum_idx+2]
        y1, y2, y3 = pdf_points[global_extremum_idx-1:global_extremum_idx+2]
        
        try:
            # Parabolic interpolation formula
            denominator = (x1 - x2) * (x1 - x3) * (x2 - x3)
            if abs(denominator) < 1e-15:
                return None
            
            A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denominator
            B = (x3*x3 * (y1 - y2) + x2*x2 * (y3 - y1) + x1*x1 * (y2 - y3)) / denominator
            
            if abs(A) < 1e-15:
                return None  # Not a proper parabola
            
            # Find vertex of parabola
            z0_candidate = -B / (2 * A)
            
            # Validate that it's a maximum and within bounds
            is_maximum = A < 0
            
            if is_maximum and x1 <= z0_candidate <= x3:
                self.estimation_info['z0_method'] = 'global_parabolic_interpolation'
                if self.verbose:
                    self.logger.info(f"Parabolic interpolation successful: Z0={z0_candidate:.8f} (PDF maximum)")
                return z0_candidate
            
        except Exception as e:
            if self.verbose:
                self.logger.error(f"Parabolic interpolation failed: {e}")
        
        return None
    
    def _get_last_method_used(self):
        return self.estimation_info.get('z0_method', 'discrete_fallback')
    
    def _get_pdf_points(self):
        """Get PDF points for E-distributions."""
        self.logger.debug("Retrieving PDF points.")
        if hasattr(self.gdf, 'pdf_points') and self.gdf.pdf_points is not None:
            return np.array(self.gdf.pdf_points)
        elif hasattr(self.gdf, 'pdf') and self.gdf.pdf is not None:
            return np.array(self.gdf.pdf)
        else:
            return np.array([])
    
    def _get_distribution_points(self):
        """Get distribution function points for Q-distributions."""
        self.logger.debug("Retrieving distribution points.")
        if hasattr(self.gdf, 'cdf_points') and self.gdf.cdf_points is not None:
            return np.array(self.gdf.cdf_points)
        elif hasattr(self.gdf, 'qgdf_points') and self.gdf.qgdf_points is not None:
            return np.array(self.gdf.qgdf_points)
        elif hasattr(self.gdf, 'qldf_points') and self.gdf.qldf_points is not None:
            return np.array(self.gdf.qldf_points)
        else:
            return np.array([])
    
    def _get_di_points(self):
        """Get data points (di_points_n) or raw data."""
        self.logger.debug("Retrieving data points.")
        if hasattr(self.gdf, 'di_points_n') and self.gdf.di_points_n is not None:
            return np.array(self.gdf.di_points_n)
        elif hasattr(self.gdf, 'data') and self.gdf.data is not None:
            # If no evaluation points, use sorted data
            return np.sort(np.array(self.gdf.data))
        else:
            return np.array([])
    
    def __repr__(self):
        target_type = "median (0.5)" if self.find_median else "PDF maximum"
        status = f"fitted (Z0={self.z0:.6f})" if self.z0 is not None else "not fitted"
        return f"Z0Estimator(gdf_type='{self.gdf_type}', target='{target_type}', {status})"