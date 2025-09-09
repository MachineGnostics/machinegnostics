"""
Z0 Estimator - Universal class for estimating Z0 point for GDF distributions

Z0 is the point where PDF reaches its global extremum (maximum for EGDF/ELDF, minimum for QLDF/QGDF),
using advanced interpolation and optimization methods.

Author: Nirmal Parmar
Machine Gnostics
"""

import numpy as np
from typing import Union, Dict, Any, Optional

class Z0Estimator:
    """
    Universal Z0 estimator for all GDF (Generalized Distribution Function) types.
    
    This class automatically detects the distribution type and finds the appropriate Z0 point:
    - For EGDF/ELDF: Finds the point where PDF reaches its global maximum
    - For QLDF/QGDF: Finds the point where PDF reaches its global minimum
    
    The estimator uses multiple advanced methods including spline optimization, polynomial fitting,
    refined interpolation, and parabolic interpolation to achieve high accuracy.
    
    Attributes:
        gdf: The fitted GDF object (EGDF, ELDF, QLDF, or QGDF)
        gdf_type (str): Detected distribution type ('egdf', 'eldf', 'qldf', 'qgdf')
        optimize (bool): Whether to use advanced optimization methods
        verbose (bool): Whether to print detailed progress information
        find_minimum (bool): True for QLDF/QGDF, False for EGDF/ELDF
        z0 (float): Estimated Z0 value (None until fit() is called)
        estimation_info (dict): Detailed information about the estimation process
    
    Examples:
        Basic usage with EGDF (finds maximum):
        >>> from machinegnostics.magcal import EGDF
        >>> egdf = EGDF(data=data)
        >>> egdf.fit()
        >>> estimator = Z0Estimator(egdf, verbose=True)
        >>> z0 = estimator.fit()
        >>> print(f"Z0 at PDF maximum: {z0}")
        
        Usage with QLDF (finds minimum):
        >>> from machinegnostics.magcal.gdf.qldf import QLDF
        >>> qldf = QLDF(data=data)
        >>> qldf.fit()
        >>> estimator = Z0Estimator(qldf, optimize=True, verbose=True)
        >>> z0 = estimator.fit()
        >>> print(f"Z0 at PDF minimum: {z0}")
        
        Simple discrete estimation (faster):
        >>> estimator = Z0Estimator(gdf_object, optimize=False)
        >>> z0 = estimator.fit()
        
        Get detailed estimation information:
        >>> info = estimator.get_estimation_info()
        >>> print(f"Method used: {info['z0_method']}")
        >>> print(f"Extremum type: {info['extremum_type']}")
        
        Visualize results:
        >>> estimator.plot_z0_analysis()
    
    Notes:
        - The GDF object must be fitted before passing to Z0Estimator
        - For flat extremum regions, the estimator finds the middle point
        - Advanced methods are tried in order of sophistication and reliability
        - The estimated Z0 is automatically assigned back to the GDF object
    """
    
    def __init__(self,
                 gdf_object,
                 optimize: bool = True,
                 verbose: bool = False):
        """
        Initialize the Z0 estimator.
        
        Args:
            gdf_object: A fitted GDF object (EGDF, ELDF, QLDF, or QGDF)
                       Must have been fitted (gdf_object.fit() called) before passing here.
            optimize (bool, optional): Whether to use advanced optimization methods.
                                     If True, uses spline optimization, polynomial fitting, etc.
                                     If False, uses simple discrete extremum finding.
                                     Defaults to True.
            verbose (bool, optional): Whether to print detailed progress information
                                    during the estimation process. Defaults to False.
        
        Raises:
            ValueError: If gdf_object is not fitted or doesn't contain required PDF data
            
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
        
        # Determine if we need to find minimum or maximum
        self.find_minimum = self.gdf_type.lower() in ['qldf', 'qgdf']
        
        # Results storage
        self.z0 = None
        self.estimation_info = {}
        
    def fit(self) -> float:
        """
        Estimate the Z0 point where PDF reaches its global extremum.
        
        For EGDF/ELDF distributions, finds the point where PDF reaches its global maximum.
        For QLDF/QGDF distributions, finds the point where PDF reaches its global minimum.
        
        The method first identifies the discrete extremum, then optionally applies advanced
        interpolation techniques for higher accuracy.
        
        Returns:
            float: The estimated Z0 value
            
        Raises:
            ValueError: If no PDF data is available for estimation
            
        Examples:
            >>> z0 = estimator.fit()
            >>> print(f"Estimated Z0: {z0:.6f}")
            
            >>> # The Z0 is automatically assigned to the GDF object
            >>> print(f"GDF Z0: {estimator.gdf.z0:.6f}")
        
        Notes:
            - For flat extremum regions, finds the middle point of the flat region
            - Advanced methods are tried in order: spline optimization, polynomial fitting,
              refined interpolation, parabolic interpolation
            - If all advanced methods fail, falls back to discrete extremum
            - The estimated Z0 is automatically assigned to the original GDF object
        """
        if self.verbose:
            operation = "minimum" if self.find_minimum else "maximum"
            print(f"Z0Estimator: Starting Z0 estimation for {self.gdf_type.upper()} (finding global {operation})")
        
        # Get PDF and data points
        pdf_points = self._get_pdf_points()
        di_points = self._get_di_points()
        
        if len(pdf_points) == 0:
            raise ValueError("No PDF data available for Z0 estimation")
        
        # Find the global extremum in the discrete data
        if self.find_minimum:
            global_extremum_idx = np.argmin(pdf_points)
            # Handle flat bottom case - find middle of minimum region
            global_extremum_idx = self._find_middle_of_flat_region(pdf_points, global_extremum_idx, find_min=True)
        else:
            global_extremum_idx = np.argmax(pdf_points)
            # Handle flat top case - find middle of maximum region
            global_extremum_idx = self._find_middle_of_flat_region(pdf_points, global_extremum_idx, find_min=False)
        
        global_extremum_value = pdf_points[global_extremum_idx]
        global_extremum_location = di_points[global_extremum_idx]
        
        if self.verbose:
            operation = "minimum" if self.find_minimum else "maximum"
            print(f"Discrete global {operation}: PDF={global_extremum_value:.6f} at x={global_extremum_location:.6f} (index {global_extremum_idx})")
        
        if self.optimize:
            self.z0 = self._find_z0_advanced(global_extremum_idx, di_points, pdf_points)
            
            if self.verbose:
                method_used = self._get_last_method_used()
                print(f"Advanced estimation complete. Method: {method_used}, Z0: {self.z0:.8f}")
        else:
            self.z0 = global_extremum_location
            
            # Store simple estimation info
            self.estimation_info = {
                'z0': self.z0,
                'z0_method': 'discrete_extremum',
                'z0_extremum_pdf_value': global_extremum_value,
                'z0_extremum_pdf_index': global_extremum_idx,
                'gdf_type': self.gdf_type,
                'extremum_type': 'minimum' if self.find_minimum else 'maximum'
            }
            
            if self.verbose:
                operation = "minimum" if self.find_minimum else "maximum"
                print(f"Simple estimation: Using discrete global {operation} at Z0={self.z0:.6f}")
        
        # Update GDF object with Z0
        self.gdf.z0 = self.z0
        if hasattr(self.gdf, 'catch') and self.gdf.catch and hasattr(self.gdf, 'params'):
            self.gdf.params['z0'] = float(self.z0)
        
        return self.z0
    
    def get_estimation_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the Z0 estimation process.
        
        Returns comprehensive information about how the Z0 value was estimated,
        including the method used, extremum type, and various diagnostic values.
        
        Returns:
            Dict[str, Any]: Dictionary containing estimation details:
                - z0 (float): The estimated Z0 value
                - z0_method (str): Method used for estimation
                - z0_extremum_pdf_value (float): PDF value at the extremum
                - z0_extremum_pdf_index (int): Array index of the extremum
                - gdf_type (str): Type of distribution ('egdf', 'eldf', 'qldf', 'qgdf')
                - extremum_type (str): Either 'minimum' or 'maximum'
                - global_extremum_idx (int): Index of the global extremum (if available)
                - global_extremum_location (float): Location of discrete extremum (if available)
                - z0_interpolation_points (int): Number of points used in interpolation (if available)
        
        Examples:
            >>> estimator.fit()
            >>> info = estimator.get_estimation_info()
            >>> print(f"Z0: {info['z0']:.6f}")
            >>> print(f"Method: {info['z0_method']}")
            >>> print(f"Extremum type: {info['extremum_type']}")
            >>> print(f"PDF value at extremum: {info['z0_extremum_pdf_value']:.6f}")
            
            >>> # Check if advanced method was used
            >>> if info['z0_method'] != 'discrete_extremum':
            >>>     print("Advanced interpolation method was successful")
        
        Notes:
            - Returns error message if fit() hasn't been called yet
            - Information varies depending on whether optimization was used
            - Useful for understanding estimation quality and method selection
        """
        if not self.estimation_info:
            return {"error": "No estimation performed yet. Call fit() first."}
        return self.estimation_info.copy()
    
    def plot_z0_analysis(self, figsize: tuple = (12, 6)) -> None:
        """
        Create visualization plots showing the Z0 estimation results.
        
        Generates a two-panel plot showing:
        1. PDF curve with the estimated Z0 point marked
        2. CDF curve (if available) or estimation information panel
        
        Args:
            figsize (tuple, optional): Figure size as (width, height) in inches.
                                     Defaults to (12, 6).
        
        Examples:
            >>> # Basic plot
            >>> estimator.fit()
            >>> estimator.plot_z0_analysis()
            
            >>> # Custom figure size
            >>> estimator.plot_z0_analysis(figsize=(15, 8))
        
        Notes:
            - Requires matplotlib to be installed
            - Must call fit() before plotting
            - Shows extremum type (minimum/maximum) in the legend
            - Red vertical line and dot mark the estimated Z0 location
            - Grid is enabled for better readability
        
        Raises:
            ImportError: If matplotlib is not available
            ValueError: If fit() hasn't been called yet
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available. Cannot create plots.")
            return
            
        if self.z0 is None:
            print("No Z0 estimation available. Call fit() first.")
            return
        
        pdf_points = self._get_pdf_points()
        di_points = self._get_di_points()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: PDF with Z0 point
        ax1.plot(di_points, pdf_points, 'b-', linewidth=2, label='PDF')
        extremum_type = "Minimum" if self.find_minimum else "Maximum"
        ax1.axvline(self.z0, color='red', linestyle='--', linewidth=2, 
                   label=f'Z0 ({extremum_type}): {self.z0:.4f}')
        ax1.scatter([self.z0], [np.interp(self.z0, di_points, pdf_points)], 
                   color='red', s=100, zorder=5)
        ax1.set_xlabel('Value')
        ax1.set_ylabel('PDF')
        ax1.set_title(f'{self.gdf_type.upper()} PDF with Z0 Point')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: CDF if available
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
            # Show estimation info instead
            info_text = f"Z0 Estimation Info:\n"
            info_text += f"Value: {self.z0:.6f}\n"
            info_text += f"Method: {self.estimation_info.get('z0_method', 'unknown')}\n"
            info_text += f"Type: {extremum_type}\n"
            info_text += f"Distribution: {self.gdf_type.upper()}"
            ax2.text(0.1, 0.5, info_text, transform=ax2.transAxes, fontsize=12,
                    verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax2.set_xlim([0, 1])
            ax2.set_ylim([0, 1])
            ax2.set_title('Z0 Estimation Information')
        
        plt.tight_layout()
        plt.show()
    
    def _find_middle_of_flat_region(self, pdf_points, extremum_idx, find_min=True):
        """Find the middle point of a flat extremum region."""
        n_points = len(pdf_points)
        extremum_value = pdf_points[extremum_idx]
        
        # Define tolerance for "flatness"
        tolerance = np.std(pdf_points) * 0.01  # 1% of standard deviation
        tolerance = max(tolerance, 1e-10)  # Minimum tolerance
        
        # Find the range of indices with similar extremum values
        if find_min:
            similar_mask = np.abs(pdf_points - extremum_value) <= tolerance
        else:
            similar_mask = np.abs(pdf_points - extremum_value) <= tolerance
        
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
                    print(f"Flat {region_type} region detected. Using middle point at index {middle_idx}")
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
            raise ValueError("GDF object must have '_fitted' attribute")
        
        if not gdf_object._fitted:
            raise ValueError("GDF object must be fitted before Z0 estimation")
        
        # Check for required PDF data
        has_pdf_points = hasattr(gdf_object, 'pdf_points') and gdf_object.pdf_points is not None
        has_pdf = hasattr(gdf_object, 'pdf') and gdf_object.pdf is not None
        
        if not (has_pdf_points or has_pdf):
            raise ValueError("GDF object must contain PDF data (pdf_points or pdf attribute)")
        
        # Check for data points
        has_di_points = hasattr(gdf_object, 'di_points_n') and gdf_object.di_points_n is not None
        has_data = hasattr(gdf_object, 'data') and gdf_object.data is not None
        
        if not (has_di_points or has_data):
            raise ValueError("GDF object must contain data points (di_points_n or data attribute)")
    
    def _detect_gdf_type(self):
        class_name = self.gdf.__class__.__name__.lower()
        
        if 'egdf' in class_name:
            return 'egdf'
        elif 'eldf' in class_name:
            return 'eldf'
        elif 'qgdf' in class_name:
            return 'qgdf'
        elif 'qldf' in class_name:
            return 'qldf'
        else:
            # Fallback - assume maximum finding for unknown types
            return 'unknown'
    
    def _find_z0_advanced(self, global_extremum_idx, di_points, pdf_points):
        """Find Z0 using advanced methods for either minimum or maximum."""
        
        # Store basic info for all methods
        extremum_value = pdf_points[global_extremum_idx]
        extremum_location = di_points[global_extremum_idx]
        
        self.estimation_info = {
            'z0': None,  # Will be updated
            'z0_method': 'discrete_extremum',  # Will be updated if advanced method succeeds
            'z0_extremum_pdf_value': extremum_value,
            'gdf_type': self.gdf_type,
            'extremum_type': 'minimum' if self.find_minimum else 'maximum',
            'global_extremum_idx': global_extremum_idx,
            'global_extremum_location': extremum_location,
            'z0_interpolation_points': len(di_points)
        }
        
        # Try advanced methods in order of preference
        advanced_methods = [
            self._try_spline_optimization,
            self._try_polynomial_fitting,
            self._try_refined_interpolation,
            self._try_parabolic_interpolation
        ]
        
        for method in advanced_methods:
            try:
                result = method(di_points, pdf_points, global_extremum_idx)
                if result is not None:
                    self.estimation_info['z0'] = result
                    return result
            except Exception as e:
                if self.verbose:
                    print(f"Method {method.__name__} failed: {e}")
                continue
        
        # All advanced methods failed - use discrete extremum
        if self.verbose:
            print("All advanced methods failed. Using discrete extremum.")
        
        self.estimation_info['z0'] = extremum_location
        return extremum_location
    
    def _try_spline_optimization(self, di_points, pdf_points, global_extremum_idx):
        try:
            from scipy.interpolate import UnivariateSpline
            from scipy.optimize import minimize_scalar
        except ImportError:
            if self.verbose:
                print("SciPy not available for spline optimization")
            return None
        
        try:
            # Create spline interpolation
            spline = UnivariateSpline(di_points, pdf_points, s=0, k=3)
            
            # Define objective function
            if self.find_minimum:
                objective = lambda x: spline(x)
            else:
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
                        operation = "minimum" if self.find_minimum else "maximum"
                        print(f"Spline optimization successful: Z0={z0_candidate:.8f} (global {operation})")
                    return z0_candidate
            
        except Exception as e:
            if self.verbose:
                print(f"Spline optimization failed: {e}")
        
        return None
    
    def _try_polynomial_fitting(self, di_points, pdf_points, global_extremum_idx):
        """Try polynomial fitting around the extremum region."""
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
                            
                            # Find the extremum
                            if self.find_minimum:
                                best_idx = np.argmin(critical_values)
                            else:
                                best_idx = np.argmax(critical_values)
                            
                            z0_candidate = valid_criticals[best_idx]
                            
                            # Validate using second derivative test
                            poly_second_deriv = np.polyder(poly_deriv)
                            second_deriv_value = poly_second_deriv(z0_candidate)
                            
                            # Check if it's the right type of extremum
                            is_minimum = second_deriv_value > 0
                            is_maximum = second_deriv_value < 0
                            
                            if (self.find_minimum and is_minimum) or (not self.find_minimum and is_maximum):
                                self.estimation_info['z0_method'] = f'global_polynomial_fitting_degree_{degree}'
                                if self.verbose:
                                    extremum_type = "minimum" if self.find_minimum else "maximum"
                                    print(f"Polynomial fitting (degree {degree}) successful: Z0={z0_candidate:.8f} ({extremum_type})")
                                return z0_candidate
                    
                    except (np.linalg.LinAlgError, ValueError) as e:
                        continue
            
        except Exception as e:
            if self.verbose:
                print(f"Polynomial fitting failed: {e}")
        
        return None
    
    def _try_refined_interpolation(self, di_points, pdf_points, global_extremum_idx):
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
            
            # Find extremum in fine grid
            if self.find_minimum:
                fine_extremum_idx = np.argmin(fine_y)
            else:
                fine_extremum_idx = np.argmax(fine_y)
            
            z0_candidate = fine_x[fine_extremum_idx]
            
            self.estimation_info['z0_method'] = 'global_refined_interpolation'
            if self.verbose:
                extremum_type = "minimum" if self.find_minimum else "maximum"
                print(f"Refined interpolation successful: Z0={z0_candidate:.8f} ({extremum_type})")
            return z0_candidate
            
        except Exception as e:
            if self.verbose:
                print(f"Refined interpolation failed: {e}")
        
        return None
    
    def _try_parabolic_interpolation(self, di_points, pdf_points, global_extremum_idx):
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
            
            # Validate that it's the right type of extremum and within bounds
            is_minimum = A > 0
            is_maximum = A < 0
            
            if ((self.find_minimum and is_minimum) or (not self.find_minimum and is_maximum)) and \
               x1 <= z0_candidate <= x3:
                
                self.estimation_info['z0_method'] = 'global_parabolic_interpolation'
                if self.verbose:
                    extremum_type = "minimum" if self.find_minimum else "maximum"
                    print(f"Parabolic interpolation successful: Z0={z0_candidate:.8f} ({extremum_type})")
                return z0_candidate
            
        except Exception as e:
            if self.verbose:
                print(f"Parabolic interpolation failed: {e}")
        
        return None
    
    def _get_last_method_used(self):
        return self.estimation_info.get('z0_method', 'discrete_extremum')
    
    def _get_pdf_points(self):
        if hasattr(self.gdf, 'pdf_points') and self.gdf.pdf_points is not None:
            return np.array(self.gdf.pdf_points)
        elif hasattr(self.gdf, 'pdf') and self.gdf.pdf is not None:
            return np.array(self.gdf.pdf)
        else:
            return np.array([])
    
    def _get_di_points(self):
        if hasattr(self.gdf, 'di_points_n') and self.gdf.di_points_n is not None:
            return np.array(self.gdf.di_points_n)
        elif hasattr(self.gdf, 'data') and self.gdf.data is not None:
            # If no evaluation points, use sorted data
            return np.sort(np.array(self.gdf.data))
        else:
            return np.array([])
    
    def __repr__(self):
        extremum_type = "minimum" if self.find_minimum else "maximum"
        status = f"fitted (Z0={self.z0:.6f})" if self.z0 is not None else "not fitted"
        return f"Z0Estimator(gdf_type='{self.gdf_type}', extremum_type='{extremum_type}', {status})"