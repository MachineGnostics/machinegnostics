"""
Sample Bound Estimator for EGDF and other GDFs.

Estimate LSB, USB for EGDF

Machine Gnostics
Author: Nirmal Parmar
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from machinegnostics.magcal.data_conversion import DataConversion

class BoundEstimator:
    """
    Class for estimating bounds for the EGDF.

    LSB (lower Sample Bound) and USB (Upper Sample Bound)

    LSB Conditions:
    Zi = transformed data to infinite domain
    1 - O < LSB < Z_min
    2 - 2nd derivative of EGDF at LSB is close to 0
    3 - 3rd derivative of EGDF at LSB is close to 0

    USB Conditions:
    1 - Z_max < USB < infinity
    2 - 2nd derivative of EGDF at USB is close to 0
    3 - 3rd derivative of EGDF at USB is close to 0
    """

    def __init__(self,
                 params: dict,
                 catch: bool = True, 
                 tolerance: float = 1e-5, 
                 distribution_type: str = 'EG',
                 verbose: bool = False):
        
        self.params = params if params is not None else {}
        self.catch = catch
        self.tolerance = tolerance
        self.lsb = None
        self.usb = None
        self.z0 = None
        self.distribution_type = distribution_type
        self.verbose = verbose
        
        # Validate inputs and extract parameters
        self._validate_inputs()
        self._extract_parameters()
        
        # Store derivatives for reuse
        self.d1 = None
        self.d2 = None
        self.d3 = None

    def _validate_inputs(self):
        """Validate input parameters."""
        if not isinstance(self.params, dict):
            raise ValueError("params must be a dictionary.")
        
        if not isinstance(self.catch, bool):
            raise ValueError("catch must be a boolean value.")
        
        if not isinstance(self.tolerance, (int, float)) or self.tolerance <= 0:
            raise ValueError("tolerance must be a positive numeric value.")
        
        if self.distribution_type not in ['EG', 'eg']:
            raise ValueError("distribution_type must be 'EG' for EGDF")
        
        if not isinstance(self.verbose, bool):
            raise ValueError("verbose must be a boolean value.")

    def _extract_parameters(self):
        """Extract and validate required parameters from params dict."""
        # Check for required EGDF data
        has_egdf_points = 'egdf_points' in self.params and 'zi_points' in self.params
        has_egdf = 'egdf' in self.params and 'zi' in self.params
        
        if not has_egdf_points and not has_egdf:
            raise ValueError("params must contain either ('egdf_points', 'zi_points') or ('egdf', 'zi').")
        
        # Extract data bounds
        self.DLB = self.params.get('DLB')
        self.DUB = self.params.get('DUB')
        self.LB = self.params.get('LB_opt')
        self.UB = self.params.get('UB_opt')
        
        # Extract data form
        self.data_form = self.params.get('data_form', 'a')

    def _get_data_egdf(self):
        """
        Get EGDF values and z_values from various sources.
        """
        egdf_values = None
        z_values = None
        
        # Priority: params['egdf_points'] > params['egdf']
        if 'egdf_points' in self.params and 'zi_points' in self.params:
            egdf_values = self.params['egdf_points']
            z_values = self.params['zi_points']
        elif 'egdf' in self.params and 'zi' in self.params:
            egdf_values = self.params['egdf']
            z_values = self.params['zi']
        else:
            raise ValueError("No EGDF data found. Provide egdf_values and z_values or set them in params.")
        
        return np.array(egdf_values), np.array(z_values)

    def _get_derivatives_egdf(self):
        """
        Calculate the derivatives 1st, 2nd, and 3rd of the EGDF.
        """
        egdf_values, z_values = self._get_data_egdf()

        egdf_1st_derivative = np.gradient(egdf_values, z_values)
        egdf_2nd_derivative = np.gradient(egdf_1st_derivative, z_values)
        egdf_3rd_derivative = np.gradient(egdf_2nd_derivative, z_values)

        return egdf_1st_derivative, egdf_2nd_derivative, egdf_3rd_derivative

    def _interpolate_derivative_egdf(self, z_target, derivative_values, z_values):
        """
        Interpolate derivative value at a specific z_target.
        """
        return np.interp(z_target, z_values, derivative_values)

    def _estimate_lsb_egdf(self):
        """
        Estimate the Lower Scale Bound (LSB) for the EGDF.
        
        LSB Conditions:
            1 - 0 < LSB < Z_min
            2 - 2nd derivative of EGDF at LSB is close to 0
            3 - 3rd derivative of EGDF at LSB is close to 0
        """
        try:
            _, z_values = self._get_data_egdf()
            z_min = np.min(z_values)
            
            # Define search range: slightly below z_min
            search_range = (z_min - 2.0, z_min - 0.01)
            
            # Objective function: minimize sum of squared 2nd and 3rd derivatives
            def objective(z):
                d2_val = self._interpolate_derivative_egdf(z, self.d2, z_values)
                d3_val = self._interpolate_derivative_egdf(z, self.d3, z_values)
                return d2_val**2 + d3_val**2
            
            # Find optimal LSB
            result = minimize_scalar(objective, bounds=search_range, method='bounded')
            
            if result.success:
                self.lsb = result.x
            else:
                # Fallback: use a point slightly before z_min
                self.lsb = z_min
                
        except Exception as e:
            if self.catch:
                # Fallback LSB estimation
                _, z_values = self._get_data_egdf()
                self.lsb = np.min(z_values)
                if self.verbose:
                    print(f"Warning: LSB estimation failed with error: {e}. Using fallback value: {self.lsb:.3f}")
            else:
                raise e

    def _estimate_usb_egdf(self):
        """
        Estimate the Upper Scale Bound (USB) for the EGDF.
        
        USB Conditions:
            1 - Z_max < USB < infinity
            2 - 2nd derivative of EGDF at USB is close to 0
            3 - 3rd derivative of EGDF at USB is close to 0
        """
        try:
            _, z_values = self._get_data_egdf()
            z_max = np.max(z_values)
            
            # Define search range: slightly above z_max
            search_range = (z_max + 0.01, z_max + 2.0)
            
            # Objective function: minimize sum of squared 2nd and 3rd derivatives
            def objective(z):
                d2_val = self._interpolate_derivative_egdf(z, self.d2, z_values)
                d3_val = self._interpolate_derivative_egdf(z, self.d3, z_values)
                return d2_val**2 + d3_val**2
            
            # Find optimal USB
            result = minimize_scalar(objective, bounds=search_range, method='bounded')
            
            if result.success:
                self.usb = result.x
            else:
                # Fallback: use a point slightly after z_max
                self.usb = z_max 

        except Exception as e:
            if self.catch:
                # Fallback USB estimation
                _, z_values = self._get_data_egdf()
                self.usb = np.max(z_values)
                if self.verbose:
                    print(f"Warning: USB estimation failed with error: {e}. Using fallback value: {self.usb:.3f}")
            else:
                raise e

    def _estimate_z0_egdf(self):
        """
        Estimate the z0 value for the EGDF.
        Location parameter where:
        1 - 1st derivative of EGDF at Z0 reaches a MAXIMUM value (PDF global maximum)
        2 - 2nd derivative of EGDF at Z0 crosses zero (inflection point)
        """
        try:
            _, z_values = self._get_data_egdf()
    
            # Method 1: Find where 1st derivative (PDF) reaches maximum
            max_d1_idx = np.argmax(self.d1)
            z0_from_max_pdf = z_values[max_d1_idx]
            
            # Method 2: Find where 2nd derivative crosses zero (inflection points)
            zero_crossings = []
            for i in range(len(self.d2) - 1):
                if self.d2[i] * self.d2[i + 1] < 0:  # Sign change indicates zero crossing
                    # Linear interpolation to find exact crossing point
                    x1, x2 = z_values[i], z_values[i + 1]
                    y1, y2 = self.d2[i], self.d2[i + 1]
                    zero_point = x1 - y1 * (x2 - x1) / (y2 - y1)
                    zero_crossings.append(zero_point)
            
            if zero_crossings:
                # Among zero crossings, find the one closest to the PDF maximum
                distances = [abs(crossing - z0_from_max_pdf) for crossing in zero_crossings]
                closest_idx = np.argmin(distances)
                z0_from_inflection = zero_crossings[closest_idx]
                
                # Use the inflection point if it's reasonably close to PDF max
                if abs(z0_from_inflection - z0_from_max_pdf) < (z_values.max() - z_values.min()) * 0.1:
                    self.z0 = z0_from_inflection
                else:
                    # Use PDF maximum if inflection point is too far
                    self.z0 = z0_from_max_pdf
            else:
                # Fallback: use PDF maximum (1st derivative maximum)
                self.z0 = z0_from_max_pdf
            
            if self.verbose:
                print(f"Z0 estimation: PDF max at {z0_from_max_pdf:.3f}, "
                      f"Inflection points at {zero_crossings}, "
                      f"Selected Z0: {self.z0:.3f}")
                
        except Exception as e:
            if self.catch:
                # Fallback: use maximum of 1st derivative (PDF maximum)
                _, z_values = self._get_data_egdf()
                max_d1_idx = np.argmax(self.d1)
                self.z0 = z_values[max_d1_idx]
                if self.verbose:
                    print(f"Warning: z0 estimation failed with error: {e}. Using PDF maximum fallback: {self.z0:.3f}")
            else:
                raise e

    def _transform_bounds_back(self):
        """Transform optimized bounds back to original domain."""
        # transform zo from infinite domain to finite domain first
        if self.z0 is not None:
            self.z0 = DataConversion._convert_inffin(self.z0, self.LB, self.UB)
        if self.data_form == 'a':
            self.lsb = DataConversion._convert_za(self.lsb, self.DLB, self.DUB)
            self.usb = DataConversion._convert_za(self.usb, self.DLB, self.DUB)
            self.z0 = DataConversion._convert_za(self.z0, self.DLB, self.DUB)
        else:
            self.lsb = DataConversion._convert_zm(self.lsb, self.DLB, self.DUB)
            self.usb = DataConversion._convert_zm(self.usb, self.DLB, self.DUB)
            self.z0 = DataConversion._convert_zm(self.z0, self.DLB, self.DUB)

    def fit(self):
        """
        Fit the bound estimator to estimate LSB, USB, and z0.
        """
        # EGDF case:
        if self.df_type.lower() == 'e' and self.df_kind.lower() == 'g':
            
            # Calculate the derivatives
            self.d1, self.d2, self.d3 = self._get_derivatives_egdf()

            # Estimate the bounds and location parameter
            self._estimate_lsb_egdf()
            self._estimate_usb_egdf()
            self._estimate_z0_egdf()

            # transform bounds to infinite domain to finite domain
            self._transform_bounds_back()

            # Store the results in params
            self.params.update({
                'LSB': self.lsb,
                'USB': self.usb,
                'Z0': self.z0,
            })
            
        else:
            raise NotImplementedError("Bound estimation for this type of GDF is not implemented yet.")

    def get_bounds(self):
        """
        Get the estimated bounds.
        
        Returns:
            dict: Dictionary containing lsb, usb, and z0 values
        """
        return {
            'LSB': self.lsb,
            'USB': self.usb,
            'Z0': self.z0
        }

    def plot(self, plot_smooth: bool = True, plot: str = 'gdf', bounds: bool = True, extra_df: bool = True, figsize=(10, 6)):
        """
        Plot EGDF using base EGDF plot style and add LSB, USB, Z0 vertical lines.
        
        Parameters:
        -----------
        plot_smooth : bool, default True
            Whether to plot smooth curves if available
        plot : str, default 'gdf'
            What to plot: 'gdf' for EGDF only, 'pdf' for PDF only, 'both' for both
        bounds : bool, default True
            Whether to display bounds on the plot
        extra_df : bool, default True
            Whether to display extra distribution functions (WEDF, KS points)
        figsize : tuple, default (10, 6)
            Figure size
        """
        try:

            import matplotlib.pyplot as plt
            
            if not self.catch:
                print("Plot is not available with argument catch=False")
                return
            
            # Validate plot parameter
            if plot not in ['gdf', 'pdf', 'both']:
                raise ValueError("plot parameter must be 'gdf', 'pdf', or 'both'")
            
            # Check required data availability
            if plot in ['gdf', 'both'] and self.params.get('egdf') is None:
                raise ValueError("EGDF must be calculated before plotting GDF")
            if plot in ['pdf', 'both'] and self.params.get('pdf') is None:
                raise ValueError("PDF must be calculated before plotting PDF")

            # Use original data points for plotting
            x_points = self.params.get('data')
            egdf_plot = self.params.get('egdf')
            pdf_plot = self.params.get('pdf')
            wedf = self.params.get('wedf')
            ksdf = self.params.get('ksdf')
            
            # Check smooth plotting availability
            has_smooth = ('di_points' in self.params and 'egdf_points' in self.params 
                        and 'pdf_points' in self.params and self.params['di_points'] is not None)
            plot_smooth = plot_smooth and has_smooth
            
            fig, ax1 = plt.subplots(figsize=figsize)
            
            # Plot EGDF (GDF) if requested
            if plot in ['gdf', 'both']:
                if plot_smooth and 'egdf_points' in self.params:
                    # Plot smooth EGDF
                    ax1.plot(x_points, egdf_plot, 'o', color='blue', label='EGDF', markersize=4)
                    ax1.plot(self.params['di_points'], self.params['egdf_points'], color='blue', 
                            linestyle='-', linewidth=2, alpha=0.8)
                else:
                    # Plot with connecting lines when smooth is False
                    ax1.plot(x_points, egdf_plot, 'o-', color='blue', label='EGDF', 
                            markersize=4, linewidth=1, alpha=0.8)
                            
                if extra_df:
                    # Plot WEDF if available
                    if wedf is not None:
                        ax1.plot(x_points, wedf, 's', color='lightblue', 
                                label='WEDF', markersize=3, alpha=0.8)
                        
                    # Plot KSDF if available
                    if ksdf is not None:
                        ax1.plot(x_points, ksdf, 's', color='cyan', 
                                label='KS Points', markersize=3, alpha=0.8)
                
                ax1.set_ylabel('EGDF', color='blue')
                ax1.tick_params(axis='y', labelcolor='blue')
                ax1.set_ylim(0, 1)
            
            # Plot PDF if requested
            if plot in ['pdf', 'both']:
                if plot == 'pdf':
                    # PDF only - use primary axis
                    if plot_smooth and 'pdf_points' in self.params:
                        # Plot smooth PDF
                        ax1.plot(x_points, pdf_plot, 'o', color='red', label='PDF', markersize=4)
                        ax1.plot(self.params['di_points'], self.params['pdf_points'], color='red', 
                                linestyle='-', linewidth=2, alpha=0.8)
                    else:
                        # Plot with connecting lines when smooth is False
                        ax1.plot(x_points, pdf_plot, 'o-', color='red', label='PDF', 
                                markersize=4, linewidth=1, alpha=0.8)
                    
                    ax1.set_ylabel('PDF', color='red')
                    ax1.tick_params(axis='y', labelcolor='red')
                    max_pdf = np.max(self.params['pdf_points']) if (plot_smooth and 'pdf_points' in self.params and self.params['pdf_points'] is not None) else np.max(pdf_plot)
                    ax1.set_ylim(0, max_pdf * 1.1)
                    ax_pdf = ax1
                else:
                    # Both - use secondary axis for PDF
                    ax2 = ax1.twinx()
                    if plot_smooth and 'pdf_points' in self.params:
                        # Plot smooth PDF
                        ax2.plot(x_points, pdf_plot, 'o', color='red', label='PDF', markersize=4)
                        ax2.plot(self.params['di_points'], self.params['pdf_points'], color='red', 
                                linestyle='-', linewidth=2, alpha=0.8)
                    else:
                        # Plot with connecting lines when smooth is False
                        ax2.plot(x_points, pdf_plot, 'o-', color='red', label='PDF', 
                                markersize=4, linewidth=1, alpha=0.8)
                    
                    ax2.set_ylabel('PDF', color='red')
                    ax2.tick_params(axis='y', labelcolor='red')
                    max_pdf = np.max(self.params['pdf_points']) if (plot_smooth and 'pdf_points' in self.params and self.params['pdf_points'] is not None) else np.max(pdf_plot)
                    ax2.set_ylim(0, max_pdf * 1.1)
                    ax_pdf = ax2
            
            # Common settings
            ax1.set_xlabel('Data Points')
            
            # Add bounds only if bounds=True
            if bounds:
                # Add original bound lines from BaseEGDF
                for bound, color, style, name in [
                    (self.params.get('DLB'), 'green', '-', 'DLB'),
                    (self.params.get('DUB'), 'orange', '-', 'DUB'),
                    (self.params.get('LB'), 'purple', '--', 'LB'),
                    (self.params.get('UB'), 'brown', '--', 'UB')
                ]:
                    if bound is not None:
                        ax1.axvline(x=bound, color=color, linestyle=style, linewidth=2, 
                                alpha=0.8, label=f"{name}={bound:.3f}")
                
                # Add NEW bound estimation lines
                if self.lsb is not None:
                    ax1.axvline(x=self.lsb, color='red', linestyle=':', linewidth=2, 
                            alpha=0.9, label=f'LSB={self.lsb:.3f}')
                
                if self.usb is not None:
                    ax1.axvline(x=self.usb, color='darkgreen', linestyle=':', linewidth=2, 
                            alpha=0.9, label=f'USB={self.usb:.3f}')
                
                if self.z0 is not None:
                    ax1.axvline(x=self.z0, color='magenta', linestyle='-', linewidth=2, 
                            alpha=0.9, label=f'Z0={self.z0:.3f}')
                
                # Add shaded regions for probable bounds (original from BaseEGDF)
                if self.params.get('LB') is not None:
                    ax1.axvspan(x_points.min(), self.params['LB'], alpha=0.15, color='purple')
                if self.params.get('UB') is not None:
                    ax1.axvspan(self.params['UB'], x_points.max(), alpha=0.15, color='brown')
            
            # Set x-axis limits
            data_range = self.params['DUB'] - self.params['DLB']
            padding = data_range * 0.1
            ax1.set_xlim(self.params['DLB'] - padding, self.params['DUB'] + padding)
            
            # Add legends
            ax1.legend(loc='upper left', bbox_to_anchor=(0, 1))
            if plot == 'both':
                ax_pdf.legend(loc='upper right', bbox_to_anchor=(1, 1))
            
            # Set title based on what's being plotted
            if plot == 'gdf':
                title = 'EGDF with Bound Estimates' + (' and Bounds' if bounds else '')
            elif plot == 'pdf':
                title = 'PDF with Bound Estimates' + (' and Bounds' if bounds else '')
            else:
                title = 'EGDF and PDF with Bound Estimates' + (' and Bounds' if bounds else '')
            
            plt.title(title)
            ax1.grid(True, alpha=0.3)
            fig.tight_layout()
            plt.show()

        except Exception as e:
            if self.catch:
                print(f"Plotting failed: {e}, check params and data availability.")
            else:
                raise e

    def summary(self):
        """
        Print a summary of the estimated bounds.
        """
        print("=== EGDF Bound Estimation Summary ===")
        print(f"Data Type: {self.df_type}GDF")
        print(f"Data Form: {self.data_form}")
        print(f"Lower Sample Bound (LSB): {self.lsb:.6f}" if self.lsb is not None else "LSB: Not estimated")
        print(f"Upper Sample Bound (USB): {self.usb:.6f}" if self.usb is not None else "USB: Not estimated")
        print(f"Location Parameter (z0): {self.z0:.6f}" if self.z0 is not None else "z0: Not estimated")
        
        if self.lsb is not None and self.usb is not None:
            print(f"Bound Range: {self.usb - self.lsb:.6f}")
        
        print("=" * 38)