"""
Machine Gnostics - Machine Gnostics Library
Copyright (C) 2025  Machine Gnostics Team

Author: Nirmal Parmar
License: GNU General Public License v3.0 (GPL-3.0)

# Sample Boundary Estimator Module
Objective: Estimate sample boundaries (LSB and USB) based on EGDF derivatives
following the theoretical framework from Chapter 15.

Terminology:
- LSB/USB: Lower/Upper Sample Boundaries (inner bounds where samples are expected)
- LB/UB: Lower/Upper Bounds (outer bounds for data transformation, LB < LSB < LP < USB < UB)

"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

class SampleBoundaryEstimator:
    """
    Estimate sample boundaries (LSB and USB) based on EGDF derivatives
    following the theoretical framework from Chapter 15.
    
    Updated to handle different data forms and search ranges:
    - Additive ('a'): LSB = data_min - data_range/2, USB = data_max + data_range/2
    - Multiplicative ('m'): LSB = data_min / sqrt(data_max/data_min), USB = data_max * sqrt(data_max/data_min)
    - None: LSB = data_min, USB = data_max (constrained search around data bounds)
    
    Terminology:
    - LSB/USB: Sample boundaries (inner bounds where samples are expected)
    - LB/UB: Outer bounds (data transformation bounds, LB < LSB < LP < USB < UB)
    """
    
    def __init__(self, z_points, egdf_cdf, data_form='a', verbose=True):
        """
        Initialize the estimator with EGDF data.
        
        Parameters:
        -----------
        z_points : array-like
            Points where EGDF is evaluated (Z_c values)
        egdf_cdf : array-like  
            PDF values of the EGDF at corresponding z_points (P(Z_c, Z_0))
        data_form : str, optional
            Data form: 'a' for additive, 'm' for multiplicative, None for general
        verbose : bool
            Whether to print detailed information
        """
        self.z_points = np.array(z_points)
        self.egdf_cdf = np.array(egdf_cdf)
        self.data_form = data_form
        self.verbose = verbose
        
        # Ensure data is sorted by z_points
        sort_idx = np.argsort(self.z_points)
        self.z_points = self.z_points[sort_idx]
        self.egdf_cdf = self.egdf_cdf[sort_idx]
        
        # Calculate initial bounds based on data form
        self._calculate_initial_bounds()
        
        if self.verbose:
            print(f"Data form: {self.data_form}")
            print(f"Data range: [{self.data_min:.6f}, {self.data_max:.6f}]")
            print(f"LSB search range: ({self.LSB_min:.6f}, {self.LSB_max:.6f})")
            print(f"USB search range: ({self.USB_min:.6f}, {self.USB_max:.6f})")
        
        # Calculate derivatives
        self._calculate_derivatives()
        
        # Create extended interpolation functions for boundary search
        self._create_extended_interpolators()
    
    def _calculate_initial_bounds(self):
        """
        Calculate initial bounds based on data form following the _initial_bounds logic.
        """
        # Data preprocessing
        self.data_min = self.z_points.min()
        self.data_max = self.z_points.max()
        self.data_range = self.data_max - self.data_min
        
        if self.data_form == 'a':
            # Additive data form
            # LSB search range: extends below data minimum
            self.LSB_min = self.data_min - self.data_range
            self.LSB_max = self.data_min
            
            # USB search range: extends above data maximum  
            self.USB_min = self.data_max
            self.USB_max = self.data_max + self.data_range
            
        elif self.data_form == 'm':
            # Multiplicative data form
            if np.any(self.z_points <= 0):
                raise ValueError("Multiplicative data must be strictly positive")
            
            # Calculate geometric bounds
            ratio_sqrt = np.sqrt(self.data_max / self.data_min)
            
            # LSB search range: geometric extension below data minimum
            self.LSB_min = self.data_min / (2 * ratio_sqrt)  # Extended lower bound
            self.LSB_max = self.data_min / ratio_sqrt
            
            # USB search range: geometric extension above data maximum
            self.USB_min = self.data_max * ratio_sqrt
            self.USB_max = self.data_max * (2 * ratio_sqrt)  # Extended upper bound
            
        elif self.data_form is None:
            # General data form - constrained search around data bounds
            # LSB search range: small extension below data minimum
            extension = self.data_range * 0.1
            self.LSB_min = self.data_min - extension
            self.LSB_max = self.data_min
            
            # USB search range: small extension above data maximum
            self.USB_min = self.data_max
            self.USB_max = self.data_max + extension
            
        else:
            raise ValueError(f"Invalid data_form: {self.data_form}. Must be 'a', 'm', or None.")
        
    def _calculate_derivatives(self):
        """
        Calculate first, second, and third derivatives of P(Z_c, Z_0) with respect to Z_0.
        Following equations 15.41 and 15.42 from the text.
        """
        # First derivative: dP/dZ_0
        self.first_deriv = np.gradient(self.egdf_cdf, self.z_points)
        
        # Second derivative: d²P/dZ_0² (equation 15.41)
        self.second_deriv = np.gradient(self.first_deriv, self.z_points)
        
        # Third derivative: d³P/dZ_0³ (equation 15.42)  
        self.third_deriv = np.gradient(self.second_deriv, self.z_points)
        
        if self.verbose:
            print("Derivatives calculated successfully")
            print(f"First derivative range: [{self.first_deriv.min()}, {self.first_deriv.max()}]")
            print(f"Second derivative range: [{self.second_deriv.min()}, {self.second_deriv.max()}]")
            print(f"Third derivative range: [{self.third_deriv.min()}, {self.third_deriv.max()}]")

    def _create_extended_interpolators(self):
        """
        Create interpolation functions that can extrapolate beyond the data range
        for searching LB and UB in extended ranges.
        """
        try:
            # Create interpolators with extrapolation capability
            self.interp_pdf = interp1d(self.z_points, self.egdf_cdf, kind='cubic', 
                                     bounds_error=False, fill_value='extrapolate')
            self.interp_first = interp1d(self.z_points, self.first_deriv, kind='cubic',
                                       bounds_error=False, fill_value='extrapolate')
            self.interp_second = interp1d(self.z_points, self.second_deriv, kind='cubic',
                                        bounds_error=False, fill_value='extrapolate')
            self.interp_third = interp1d(self.z_points, self.third_deriv, kind='cubic',
                                       bounds_error=False, fill_value='extrapolate')
            
            if self.verbose:
                print("Extended interpolators created successfully")
                
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not create cubic interpolators ({e})")
                print("Falling back to linear interpolation")
            
            # Fallback to linear interpolation
            self.interp_pdf = interp1d(self.z_points, self.egdf_cdf, kind='linear',
                                     bounds_error=False, fill_value='extrapolate')
            self.interp_first = interp1d(self.z_points, self.first_deriv, kind='linear',
                                       bounds_error=False, fill_value='extrapolate')
            self.interp_second = interp1d(self.z_points, self.second_deriv, kind='linear',
                                        bounds_error=False, fill_value='extrapolate')
            self.interp_third = interp1d(self.z_points, self.third_deriv, kind='linear',
                                       bounds_error=False, fill_value='extrapolate')

    def find_inflection_points_in_ranges(self, n_points=1000):
        """
        Find inflection points where second derivative changes sign within the specified ranges.
        """
        # Create extended z-axis for searching
        LSB_search_points = np.linspace(self.LSB_min + 1e-6, self.LSB_max - 1e-6, n_points//2)
        USB_search_points = np.linspace(self.USB_min + 1e-6, self.USB_max - 1e-6, n_points//2)
        
        # Evaluate second derivative in search ranges
        LSB_second_deriv = self.interp_second(LSB_search_points)
        USB_second_deriv = self.interp_second(USB_search_points)
        
        # Find zero crossings in LSB range
        LSB_inflections = []
        LSB_sign_changes = np.diff(np.sign(LSB_second_deriv))
        LSB_zero_crossings = np.where(np.abs(LSB_sign_changes) > 0)[0]
        
        for idx in LSB_zero_crossings:
            if idx < len(LSB_search_points) - 1:
                z1, z2 = LSB_search_points[idx], LSB_search_points[idx + 1]
                d1, d2 = LSB_second_deriv[idx], LSB_second_deriv[idx + 1]
                
                if abs(d2 - d1) > 1e-12:
                    z_zero = z1 - d1 * (z2 - z1) / (d2 - d1)
                    if self.LSB_min < z_zero < self.LSB_max:
                        LSB_inflections.append(z_zero)
        
        # Find zero crossings in USB range
        USB_inflections = []
        USB_sign_changes = np.diff(np.sign(USB_second_deriv))
        USB_zero_crossings = np.where(np.abs(USB_sign_changes) > 0)[0]
        
        for idx in USB_zero_crossings:
            if idx < len(USB_search_points) - 1:
                z1, z2 = USB_search_points[idx], USB_search_points[idx + 1]
                d1, d2 = USB_second_deriv[idx], USB_second_deriv[idx + 1]
                
                if abs(d2 - d1) > 1e-12:
                    z_zero = z1 - d1 * (z2 - z1) / (d2 - d1)
                    if self.USB_min < z_zero < self.USB_max:
                        USB_inflections.append(z_zero)
        
        return np.array(LSB_inflections), np.array(USB_inflections)

    def estimate_boundaries_theoretical(self, tolerance=1e-4, n_points=1000):
        """
        Estimate LSB and USB using the theoretical approach:
        Find points where both second and third derivatives are zero within specified ranges.
        
        This implements the conditions from equations 15.41 and 15.42.
        """
        # Create search grids in the specified ranges
        LSB_search_points = np.linspace(self.LSB_min + 1e-6, self.LSB_max - 1e-6, n_points//2)
        USB_search_points = np.linspace(self.USB_min + 1e-6, self.USB_max - 1e-6, n_points//2)
        
        # Evaluate derivatives at search points
        LSB_second = self.interp_second(LSB_search_points)
        LSB_third = self.interp_third(LSB_search_points)
        USB_second = self.interp_second(USB_search_points)
        USB_third = self.interp_third(USB_search_points)
        
        # Find LSB candidates
        LSB_second_mask = np.abs(LSB_second) < tolerance
        LSB_third_mask = np.abs(LSB_third) < tolerance
        LSB_combined_mask = LSB_second_mask & LSB_third_mask
        LSB_candidates = LSB_search_points[LSB_combined_mask]
        
        # Find USB candidates
        USB_second_mask = np.abs(USB_second) < tolerance
        USB_third_mask = np.abs(USB_third) < tolerance
        USB_combined_mask = USB_second_mask & USB_third_mask
        USB_candidates = USB_search_points[USB_combined_mask]
        
        # Select best candidates
        LSB = LSB_candidates[0] if len(LSB_candidates) > 0 else None
        USB = USB_candidates[-1] if len(USB_candidates) > 0 else None
        
        # If no candidates found, try with relaxed tolerance
        if LSB is None or USB is None:
            tolerance *= 10
            if self.verbose:
                print(f"No candidates found, relaxing tolerance to {tolerance}")
            
            if LSB is None:
                LSB_second_mask = np.abs(LSB_second) < tolerance
                LSB_third_mask = np.abs(LSB_third) < tolerance
                LSB_combined_mask = LSB_second_mask & LSB_third_mask
                LSB_candidates = LSB_search_points[LSB_combined_mask]
                LSB = LSB_candidates[0] if len(LSB_candidates) > 0 else None
            
            if USB is None:
                USB_second_mask = np.abs(USB_second) < tolerance
                USB_third_mask = np.abs(USB_third) < tolerance
                USB_combined_mask = USB_second_mask & USB_third_mask
                USB_candidates = USB_search_points[USB_combined_mask]
                USB = USB_candidates[-1] if len(USB_candidates) > 0 else None
        
        if self.verbose:
            print(f"LSB candidates in range ({self.LSB_min}, {self.LSB_max}): {len(LSB_candidates) if LSB is not None else 0}")
            print(f"USB candidates in range ({self.USB_min}, {self.USB_max}): {len(USB_candidates) if USB is not None else 0}")
            if LSB is not None:
                print(f"Selected LSB: {LSB}")
            if USB is not None:
                print(f"Selected USB: {USB}")
                
        return LSB, USB

    def estimate_boundaries_inflection(self):
        """
        Alternative method: Use inflection points of the density function within specified ranges.
        """
        LSB_inflections, USB_inflections = self.find_inflection_points_in_ranges()
        
        # Select LSB and USB from inflection points
        LSB = LSB_inflections[0] if len(LSB_inflections) > 0 else None
        USB = USB_inflections[-1] if len(USB_inflections) > 0 else None
        
        # Fallback if no inflection points found
        if LSB is None:
            LSB = (self.LSB_min + self.LSB_max) / 2
            
        if USB is None:
            USB = (self.USB_min + self.USB_max) / 2
            
        if self.verbose:
            print(f"Inflection-based estimation:")
            print(f"Found {len(LSB_inflections)} LSB inflection points: {LSB_inflections}")
            print(f"Found {len(USB_inflections)} USB inflection points: {USB_inflections}")
            print(f"Selected LSB: {LSB}, USB: {USB}")
            
        return LSB, USB

    def estimate_boundaries_optimization(self):
        """
        Numerical optimization approach to find points where derivatives are minimized
        within the specified ranges.
        """
        def objective_LSB(z):
            """Objective function for LSB: sum of squares of second and third derivatives"""
            return self.interp_second(z)**2 + self.interp_third(z)**2
        
        def objective_USB(z):
            """Objective function for USB: sum of squares of second and third derivatives"""
            return self.interp_second(z)**2 + self.interp_third(z)**2
        
        # Find LSB using optimization
        LSB = None
        try:
            LSB_result = minimize_scalar(objective_LSB, bounds=(self.LSB_min + 1e-6, self.LSB_max - 1e-6), 
                                       method='bounded')
            if LSB_result.success and LSB_result.fun < 1e-3:
                LSB = LSB_result.x
        except Exception as e:
            if self.verbose:
                print(f"LSB optimization failed: {e}")
        
        # Find USB using optimization
        USB = None
        try:
            USB_result = minimize_scalar(objective_USB, bounds=(self.USB_min + 1e-6, self.USB_max - 1e-6), 
                                       method='bounded')
            if USB_result.success and USB_result.fun < 1e-3:
                USB = USB_result.x
        except Exception as e:
            if self.verbose:
                print(f"USB optimization failed: {e}")
        
        # Fallback values if optimization fails
        if LSB is None:
            LSB = (self.LSB_min + self.LSB_max) / 2
            
        if USB is None:
            USB = (self.USB_min + self.USB_max) / 2
            
        if self.verbose:
            print(f"Optimization-based estimation:")
            print(f"LSB optimization {'successful' if LSB != (self.LSB_min + self.LSB_max) / 2 else 'failed'}")
            print(f"USB optimization {'successful' if USB != (self.USB_min + self.USB_max) / 2 else 'failed'}")
            print(f"Selected LSB: {LSB}, USB: {USB}")
            
        return LSB, USB

    def estimate_all_methods(self):
        """
        Apply all three estimation methods and return results.
        """
        results = {}
        
        print("=" * 60)
        print(f"SAMPLE BOUNDARY ESTIMATION (DATA FORM: {self.data_form})")
        print("=" * 60)
        print(f"Data form: {self.data_form}")
        print(f"Data range: [{self.data_min:.6f}, {self.data_max:.6f}]")
        print(f"LB search range: ({self.LB_min:.6f}, {self.LB_max:.6f})")
        print(f"UB search range: ({self.UB_min:.6f}, {self.UB_max:.6f})")
        
        # Method 1: Theoretical approach (equations 15.41, 15.42)
        print("\n1. THEORETICAL METHOD (Equations 15.41 & 15.42):")
        print("-" * 50)
        LB_theo, UB_theo = self.estimate_boundaries_theoretical()
        results['theoretical'] = {'LB': LB_theo, 'UB': UB_theo}
        
        # Method 2: Inflection points method  
        print("\n2. INFLECTION POINTS METHOD:")
        print("-" * 35)
        LB_infl, UB_infl = self.estimate_boundaries_inflection()
        results['inflection'] = {'LB': LB_infl, 'UB': UB_infl}
        
        # Method 3: Optimization method
        print("\n3. OPTIMIZATION METHOD:")
        print("-" * 25)
        LB_opt, UB_opt = self.estimate_boundaries_optimization()
        results['optimization'] = {'LB': LB_opt, 'UB': UB_opt}
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY OF RESULTS:")
        print("=" * 60)
        for method, bounds in results.items():
            if bounds['LB'] is not None:
                lb_str = f"{bounds['LB']:.6f}"
            else:
                lb_str = "None (failed)"
                
            if bounds['UB'] is not None:
                ub_str = f"{bounds['UB']:.6f}"
            else:
                ub_str = "None (failed)"
                
            print(f"{method.upper()}: LB = {lb_str}, UB = {ub_str}")
        
        # Data form specific interpretation
        print(f"\nDATA FORM INTERPRETATION ({self.data_form}):")
        print("-" * 40)
        if self.data_form == 'a':
            print("• Additive data: Boundaries represent linear extensions beyond data range")
            print("• LB/UB suitable for symmetric distributions around data center")
        elif self.data_form == 'm':
            print("• Multiplicative data: Boundaries represent geometric extensions")
            print("• LB/UB account for logarithmic scaling and positive constraint")
        elif self.data_form is None:
            print("• General data: Conservative boundaries close to data extremes")
            print("• LB/UB provide minimal extension for robust estimation")
            
        return results

    def plot_analysis(self, figsize=(15, 12)):
        """
        Create comprehensive plots showing the analysis including search ranges with peak/valley detection.
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Create extended z-axis for plotting search ranges
        z_extended = np.linspace(self.LB_min, self.UB_max, 1000)
        
        # Find peaks and valleys in the data
        pdf_peaks, _ = find_peaks(self.egdf_cdf, height=0.01, distance=5)
        pdf_valleys, _ = find_peaks(-self.egdf_cdf, height=-0.01, distance=5)
        
        first_peaks, _ = find_peaks(self.first_deriv, height=0.01, distance=5)
        first_valleys, _ = find_peaks(-self.first_deriv, height=-0.01, distance=5)
        
        second_peaks, _ = find_peaks(self.second_deriv, height=0.01, distance=5)
        second_valleys, _ = find_peaks(-self.second_deriv, height=-0.01, distance=5)
        
        third_peaks, _ = find_peaks(self.third_deriv, height=0.01, distance=5)
        third_valleys, _ = find_peaks(-self.third_deriv, height=-0.01, distance=5)
        
        # Plot 1: Original PDF with search ranges and peaks/valleys
        axes[0,0].plot(self.z_points, self.egdf_cdf, 'b-', linewidth=3, label='EGDF')
        
        # Mark peaks and valleys
        if len(pdf_peaks) > 0:
            axes[0,0].plot(self.z_points[pdf_peaks], self.egdf_cdf[pdf_peaks], 
                          'go', markersize=8, label='Peaks', zorder=6)
        if len(pdf_valleys) > 0:
            axes[0,0].plot(self.z_points[pdf_valleys], self.egdf_cdf[pdf_valleys], 
                          'ro', markersize=8, label='Valleys', zorder=6)
        
        axes[0,0].axvspan(self.LB_min, self.LB_max, alpha=0.2, color='red', label='LB Search Range')
        axes[0,0].axvspan(self.UB_min, self.UB_max, alpha=0.2, color='green', label='UB Search Range')
        axes[0,0].axvline(self.data_min, color='blue', linestyle='--', alpha=0.7, label='Data Min')
        axes[0,0].axvline(self.data_max, color='blue', linestyle='--', alpha=0.7, label='Data Max')
        axes[0,0].set_title(f'EGDF with Boundary Search Ranges (Data Form: {self.data_form})')
        axes[0,0].set_xlabel('Z_c')
        axes[0,0].set_ylabel('P(Z_c, Z_0)')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].legend()
        
        # Plot 2: First derivative with peaks/valleys
        axes[0,1].plot(self.z_points, self.first_deriv, 'g-', linewidth=3, label="dP/dZ_0")
        
        # Mark peaks and valleys
        if len(first_peaks) > 0:
            axes[0,1].plot(self.z_points[first_peaks], self.first_deriv[first_peaks], 
                          'mo', markersize=8, label='Peaks', zorder=6)
        if len(first_valleys) > 0:
            axes[0,1].plot(self.z_points[first_valleys], self.first_deriv[first_valleys], 
                          'ro', markersize=8, label='Valleys', zorder=6)
        
        axes[0,1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[0,1].axvspan(self.LB_min, self.LB_max, alpha=0.2, color='red')
        axes[0,1].axvspan(self.UB_min, self.UB_max, alpha=0.2, color='green')
        axes[0,1].set_title('First Derivative (PDF)')
        axes[0,1].set_xlabel('Z_c')
        axes[0,1].set_ylabel("dP/dZ_0")
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].legend()
        
        # Plot 3: Second derivative (equation 15.41) with peaks/valleys
        axes[1,0].plot(self.z_points, self.second_deriv, 'r-', linewidth=3, label="d²P/dZ_0²")
        
        # Mark peaks and valleys
        if len(second_peaks) > 0:
            axes[1,0].plot(self.z_points[second_peaks], self.second_deriv[second_peaks], 
                          'go', markersize=8, label='Peaks', zorder=6)
        if len(second_valleys) > 0:
            axes[1,0].plot(self.z_points[second_valleys], self.second_deriv[second_valleys], 
                          'mo', markersize=8, label='Valleys', zorder=6)
        
        axes[1,0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1,0].axvspan(self.LB_min, self.LB_max, alpha=0.2, color='red')
        axes[1,0].axvspan(self.UB_min, self.UB_max, alpha=0.2, color='green')
        axes[1,0].set_title('Second Derivative')
        axes[1,0].set_xlabel('Z_c')
        axes[1,0].set_ylabel("d²P/dZ_0²")
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].legend()
        
        # Plot 4: Third derivative (equation 15.42) with peaks/valleys
        axes[1,1].plot(self.z_points, self.third_deriv, 'm-', linewidth=3, label="d³P/dZ_0³")
        
        # Mark peaks and valleys
        if len(third_peaks) > 0:
            axes[1,1].plot(self.z_points[third_peaks], self.third_deriv[third_peaks], 
                          'go', markersize=8, label='Peaks', zorder=6)
        if len(third_valleys) > 0:
            axes[1,1].plot(self.z_points[third_valleys], self.third_deriv[third_valleys], 
                          'ro', markersize=8, label='Valleys', zorder=6)
        
        axes[1,1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1,1].axvspan(self.LB_min, self.LB_max, alpha=0.2, color='red')
        axes[1,1].axvspan(self.UB_min, self.UB_max, alpha=0.2, color='green')
        axes[1,1].set_title('Third Derivative')
        axes[1,1].set_xlabel('Z_c')
        axes[1,1].set_ylabel("d³P/dZ_0³")
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].legend()
        
        # Add boundary estimates to plots
        try:
            results = self.estimate_all_methods()
            colors = ['blue', 'red', 'green']
            methods = ['theoretical', 'inflection', 'optimization']
            
            for i, (method, color) in enumerate(zip(methods, colors)):
                LB = results[method]['LB']
                UB = results[method]['UB']
                
                if LB is not None and UB is not None:
                    for ax in axes.flat:
                        ax.axvline(x=LB, color=color, linestyle=':', linewidth=2, alpha=0.8, 
                                 label=f'{method} LB' if ax == axes[0,0] else "")
                        ax.axvline(x=UB, color=color, linestyle=':', linewidth=2, alpha=0.8,
                                 label=f'{method} UB' if ax == axes[0,0] else "")
        except Exception as e:
            if self.verbose:
                print(f"Error adding boundary estimates to plot: {e}")
            
        axes[0,0].legend()
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def _find_zeros(self, x, y, tolerance=1e-6):
        """Helper method to find zero crossings in a function."""
        zeros = []
        for i in range(len(y) - 1):
            if y[i] * y[i+1] <= 0 and abs(y[i] - y[i+1]) > tolerance:
                # Linear interpolation to find more precise zero
                x_zero = x[i] - y[i] * (x[i+1] - x[i]) / (y[i+1] - y[i])
                zeros.append(x_zero)
        return zeros
    
    def _print_summary_statistics(self, results):
        """Print detailed summary statistics."""
        print(f"Data Form: {self.data_form or 'General'}")
        print(f"Data Range: [{self.data_min:.6f}, {self.data_max:.6f}] (width: {self.data_range:.6f})")
        print(f"Search Ranges: LB[{self.LB_min:.6f}, {self.LB_max:.6f}], UB[{self.UB_min:.6f}, {self.UB_max:.6f}]")
        print()
        
        # Method results
        for method in ['theoretical', 'inflection', 'optimization']:
            lb = results[method]['LB']
            ub = results[method]['UB']
            if lb is not None and ub is not None:
                width = ub - lb
                print(f"{method.title()} Method: LB={lb:.6f}, UB={ub:.6f}, Width={width:.6f}")
            else:
                print(f"{method.title()} Method: Failed to find boundaries")
        
        # Agreement analysis
        lb_vals = [results[m]['LB'] for m in ['theoretical', 'inflection', 'optimization'] 
                  if results[m]['LB'] is not None]
        ub_vals = [results[m]['UB'] for m in ['theoretical', 'inflection', 'optimization'] 
                  if results[m]['UB'] is not None]
        
        if len(lb_vals) >= 2:
            lb_agreement = np.std(lb_vals) / np.mean(np.abs(lb_vals)) * 100
            ub_agreement = np.std(ub_vals) / np.mean(np.abs(ub_vals)) * 100
            print(f"\nMethod Agreement: LB={lb_agreement:.2f}% CV, UB={ub_agreement:.2f}% CV")
        
        print("="*80)

# print("Updated SampleBoundaryEstimator class defined successfully!")
# print("Key updates:")
# print("- Data form support: 'a' (additive), 'm' (multiplicative), None (general)")
# print("- Additive: LB = data_min - data_range, UB = data_max + data_range")
# print("- Multiplicative: LB/UB calculated using geometric extensions")
# print("- General: Conservative boundaries with minimal extension")
# print("- Extended interpolation for boundary search beyond data range")
# print("- Constrained optimization within data-form-specific ranges")
# print("- Enhanced visualization showing search ranges and data form")