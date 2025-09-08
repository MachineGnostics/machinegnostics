'''
QGDF: Quantifying Global Distribution Functions

Author: Nirmal Parmar
Machine Gnostics
'''
import numpy as np
import warnings
from machinegnostics.magcal.gdf.base_distfunc import BaseDistFuncCompute
from machinegnostics.magcal.data_conversion import DataConversion
from machinegnostics.magcal.characteristics import GnosticsCharacteristics

class BaseQGDF(BaseDistFuncCompute):
    """
    Base class for Quantifying Global Distribution Functions (QGDF).
    
    This class provides foundational methods and attributes for computing
    and analyzing global distribution functions using various techniques.
    
    Attributes:
        data (np.ndarray): Input data for distribution function computation.
        n_points (int): Number of points for evaluation.
        S (float): Smoothing parameter.
        catch (bool): Flag to enable error catching.
        verbose (bool): Flag to enable verbose output.
        params (dict): Dictionary to store parameters and results.
    """
    
    def __init__(self,
                 data: np.ndarray,
                 DLB: float = None,
                 DUB: float = None,
                 LB: float = None,
                 UB: float = None,
                 S = 'auto',
                 z0_optimize: bool = True,
                 tolerance: float = 1e-3,
                 data_form: str = 'a',
                 n_points: int = 500,
                 homogeneous: bool = True,
                 catch: bool = True,
                 weights: np.ndarray = None,
                 wedf: bool = True,
                 opt_method: str = 'L-BFGS-B',
                 verbose: bool = False,
                 max_data_size: int = 1000,
                 flush: bool = True):
        super().__init__(data=data, 
                         DLB=DLB, 
                         DUB=DUB, 
                         LB=LB, 
                         UB=UB, 
                         S=S, 
                         z0_optimize=z0_optimize,
                         varS=False, # NOTE for QGDFF varS is always False 
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

        # Store raw inputs
        self.data = data
        self.DLB = DLB
        self.DUB = DUB
        self.LB = LB
        self.UB = UB
        self.S = S
        self.z0_optimize = z0_optimize

        self.tolerance = tolerance
        self.data_form = data_form
        self.n_points = n_points
        self.homogeneous = homogeneous
        self.catch = catch
        self.weights = weights if weights is not None else np.ones_like(data)
        self.wedf = wedf
        self.opt_method = opt_method
        self.verbose = verbose
        self.max_data_size = max_data_size
        self.flush = flush
        
        # Initialize state variables
        self.params = {}
        self._fitted = False
        self._derivatives_calculated = False
        self._marginal_analysis_done = False
        
        # Initialize computation cache
        self._computation_cache = {
            'data_converter': None,
            'characteristics_computer': None,
            'weights_normalized': None,
            'smooth_curves_generated': False
        }
        
        # Store initial parameters if catching
        if self.catch:
            self._store_initial_params()

        # Validate all inputs
        self._validate_inputs()

    def _compute_qgdf_core(self, S, LB, UB, zi_data=None, zi_eval=None):
        """Core QGDF computation with caching."""
        # Use provided data or default to instance data
        if zi_data is None:
            zi_data = self.z
        if zi_eval is None:
            zi_eval = zi_data
        
        # Convert to infinite domain
        zi_n = DataConversion._convert_fininf(zi_eval, LB, UB)
        zi_d = DataConversion._convert_fininf(zi_data, LB, UB)
        
        # Calculate R matrix with numerical stability
        R = zi_n.reshape(-1, 1) / (zi_d.reshape(1, -1) + self._NUMERICAL_EPS)
        
        # Get characteristics
        gc = GnosticsCharacteristics(R=R)
        q, q1 = gc._get_q_q1(S=S)
        
        # Calculate fidelities and irrelevances
        fj = gc._fj(q=q, q1=q1)
        hj = gc._hj(q=q, q1=q1)
        
        # Estimate QGDF
        return self._estimate_qgdf_from_moments(fj, hj), fj, hj

    def _estimate_qgdf_from_moments(self, fidelities, irrelevances):
        """Estimate QGDF from fidelities and irrelevances using equation (15.37)."""
        weights = self._computation_cache['weights_normalized'].reshape(-1, 1)
        
        # Calculate weighted means (f̄Q and h̄Q from equation 15.35)
        mean_fidelity = np.sum(weights * fidelities, axis=0) / np.sum(weights)  # f̄Q
        mean_irrelevance = np.sum(weights * irrelevances, axis=0) / np.sum(weights)  # h̄Q
        
        # Check for overflow potential BEFORE squaring
        sqrt_max = np.sqrt(np.finfo(float).max)
        fidelity_too_large = np.abs(mean_fidelity) > sqrt_max
        irrelevance_too_large = np.abs(mean_irrelevance) > sqrt_max
        
        if np.any(fidelity_too_large) or np.any(irrelevance_too_large):
            if self.verbose:
                print("Warning: Very large fidelity/irrelevance values detected. Using fallback calculation.")
            # Fallback: use direct ratio approach without squaring
            mean_fidelity_safe = np.where(np.abs(mean_fidelity) < self._NUMERICAL_EPS, 
                                        self._NUMERICAL_EPS, mean_fidelity)
            qgdf_values = (1 - mean_irrelevance / mean_fidelity_safe) / 2
        else:
            # Safe to square - no overflow will occur
            fidelity_squared = mean_fidelity**2
            irrelevance_squared = mean_irrelevance**2
            
            # Calculate hZ,j using equation (15.35): hZ,j = h̄Q / sqrt((f̄Q)² - (h̄Q)²)
            # Check if (f̄Q)² - (h̄Q)² is positive to avoid complex numbers
            diff_squared = fidelity_squared - irrelevance_squared
            
            # Only proceed if we have valid (positive) values under the square root
            valid_mask = diff_squared > 0
            
            if not np.any(valid_mask):
                if self.verbose:
                    print("Warning: All values lead to invalid square root. Using fallback calculation.")
                # Fallback: use direct ratio approach
                mean_fidelity_safe = np.where(np.abs(mean_fidelity) < self._NUMERICAL_EPS, 
                                            self._NUMERICAL_EPS, mean_fidelity)
                qgdf_values = (1 - mean_irrelevance / mean_fidelity_safe) / 2
            else:
                # Standard calculation where valid
                denominator = np.sqrt(np.maximum(diff_squared, self._NUMERICAL_EPS))
                h_zj = np.where(valid_mask, mean_irrelevance / denominator, 0)
                
                # Calculate hGQ using equation (15.36): hGQ = hZ,j / sqrt(1 + (hZ,j)²)
                # Check for overflow in h_zj squaring
                if np.any(np.abs(h_zj) > sqrt_max):
                    if self.verbose:
                        print("Warning: Large h_zj values. Using approximation.")
                    # For very large h_zj, h_gq approaches sign(h_zj)
                    h_gq = np.sign(h_zj)
                else:
                    h_zj_squared = h_zj**2
                    h_gq = h_zj / np.sqrt(1 + h_zj_squared)
                
                # Calculate QGDF using equation (15.37): QGDF = (1 - h̄Q/f̄Q) / 2
                mean_fidelity_safe = np.where(np.abs(mean_fidelity) < self._NUMERICAL_EPS, 
                                            self._NUMERICAL_EPS, mean_fidelity)
                qgdf_values = (1 - mean_irrelevance / mean_fidelity_safe) / 2
        
        # Ensure monotonic and bounded
        qgdf_values = np.maximum.accumulate(qgdf_values)
        qgdf_values = np.clip(qgdf_values, 0, 1)
        
        return qgdf_values.flatten()
    
    def _calculate_pdf_from_moments(self, fidelities, irrelevances):
        """Calculate PDF from fidelities and irrelevances using equation (15.38)."""
        if fidelities is None or irrelevances is None:
            raise ValueError("Fidelities and irrelevances must be calculated first")
        
        weights = self._computation_cache['weights_normalized'].reshape(-1, 1)
        
        # Calculate weighted means
        mean_fidelity = np.sum(weights * fidelities, axis=0) / np.sum(weights)  # f̄Q
        mean_irrelevance = np.sum(weights * irrelevances, axis=0) / np.sum(weights)  # h̄Q
        
        # Second order moments for derivatives (scaled by S)
        fidelities_squared = np.clip(fidelities**2, 0, np.finfo(float).max)
        f2s = np.sum(weights * (fidelities_squared / self.S_opt), axis=0) / np.sum(weights)
        
        # Calculate PDF using equation (15.38): dQG/dZ0 = (1/SZ0) * (1 - (h̄Q)²/(f̄Q)²)
        mean_fidelity_safe = np.where(np.abs(mean_fidelity) < self._NUMERICAL_EPS, 
                                     self._NUMERICAL_EPS, mean_fidelity)
        
        # Calculate the ratio and its square with overflow protection
        ratio = mean_irrelevance / mean_fidelity_safe
        ratio_squared = np.clip(ratio**2, 0, np.finfo(float).max)
        
        # Calculate the term (1 - (h̄Q)²/(f̄Q)²)
        pdf_term = 1 - ratio_squared
        
        # Ensure PDF term is non-negative (mathematical requirement)
        pdf_term = np.maximum(pdf_term, 0)
        
        # Apply the scaling factor (1/SZ0) - using f2s as proxy for 1/Z0 derivative scaling
        S_value = self.S_opt if hasattr(self, 'S_opt') else 1.0
        f2s_clipped = np.clip(f2s, 0, np.finfo(float).max)
        
        pdf_values = (1 / S_value) * pdf_term * f2s_clipped
        
        # Final clipping to ensure numerical stability
        pdf_values = np.clip(pdf_values, 0, np.finfo(float).max)
        
        return pdf_values.flatten()

    def _calculate_final_results(self):
        """Calculate final QGDF and PDF with optimized parameters."""
        # Convert to infinite domain
        # zi_n = DataConversion._convert_fininf(self.z, self.LB_opt, self.UB_opt)
        zi_d = DataConversion._convert_fininf(self.z, self.LB_opt, self.UB_opt)
        self.zi = zi_d

        # Calculate QGDF and get moments
        qgdf_values, fi, hi = self._compute_qgdf_core(self.S_opt, self.LB_opt, self.UB_opt)

        # Store for derivative calculations
        self.fi = fi
        self.hi = hi
        self.qgdf = qgdf_values
        self.pdf = self._calculate_pdf_from_moments(fi, hi)
        
        if self.catch:
            self.params.update({
                'qgdf': self.qgdf.copy(),
                'pdf': self.pdf.copy(),
                'zi': self.zi.copy()
            })

    def _generate_smooth_curves(self):
        """Generate smooth curves for plotting and analysis."""
        try:
            # Generate smooth QGDF and PDF
            smooth_qgdf, self.smooth_fj, self.smooth_hj = self._compute_qgdf_core(
                self.S_opt, self.LB_opt, self.UB_opt,
                zi_data=self.z_points_n, zi_eval=self.z
            )
            
            smooth_pdf = self._calculate_pdf_from_moments(self.smooth_fj, self.smooth_hj)

            self.qgdf_points = smooth_qgdf
            self.pdf_points = smooth_pdf
            
            # Store zi_n for derivative calculations
            self.zi_n = DataConversion._convert_fininf(self.z_points_n, self.LB_opt, self.UB_opt)
            
            # Mark as generated
            self._computation_cache['smooth_curves_generated'] = True
            
            if self.catch:
                self.params.update({
                    'qgdf_points': self.qgdf_points.copy(),
                    'pdf_points': self.pdf_points.copy(),
                    'zi_points': self.zi_n.copy()
                })
            
            if self.verbose:
                print(f"Generated smooth curves with {self.n_points} points.")
                
        except Exception as e:
            # Log the error
            error_msg = f"Could not generate smooth curves: {e}"
            self.params['errors'].append({
                'method': '_generate_smooth_curves',
                'error': error_msg,
                'exception_type': type(e).__name__
            })
            if self.verbose:
                print(f"Warning: Could not generate smooth curves: {e}")
            # Create fallback points using original data
            self.qgdf_points = self.qgdf.copy() if hasattr(self, 'qgdf') else None
            self.pdf_points = self.pdf.copy() if hasattr(self, 'pdf') else None
            self._computation_cache['smooth_curves_generated'] = False

    def _get_results(self)-> dict:
        """Return fitting results."""
        if not self._fitted:
            raise RuntimeError("Must fit QGDF before getting results.")

        # selected key from params if exists
        keys = ['DLB', 'DUB', 'LB', 'UB', 'S_opt', 'z0', 'qgdf', 'pdf', 
                'qgdf_points', 'pdf_points', 'zi', 'zi_points', 'weights']
        results = {key: self.params.get(key) for key in keys if key in self.params}
        return results


    def _plot(self, plot_smooth: bool = True, plot: str = 'both', bounds: bool = True, extra_df: bool = True, figsize: tuple = (12, 8)):
        """Enhanced plotting with better organization."""
        import matplotlib.pyplot as plt

        if plot_smooth and (len(self.data) > self.max_data_size) and self.verbose:
            print(f"Warning: Given data size ({len(self.data)}) exceeds max_data_size ({self.max_data_size}). For optimal compute performance, set 'plot_smooth=False', or 'max_data_size' to a larger value whichever is appropriate.")

        if not self.catch:
            print("Plot is not available with argument catch=False")
            return
        
        if not self._fitted:
            raise RuntimeError("Must fit QGDF before plotting.")
        
        # Validate plot parameter
        if plot not in ['gdf', 'pdf', 'both']:
            raise ValueError("plot parameter must be 'gdf', 'pdf', or 'both'")
        
        # Check data availability
        if plot in ['gdf', 'both'] and self.params.get('qgdf') is None:
            raise ValueError("QGDF must be calculated before plotting GDF")
        if plot in ['pdf', 'both'] and self.params.get('pdf') is None:
            raise ValueError("PDF must be calculated before plotting PDF")
        
        # Prepare data
        x_points = self.data
        qgdf_plot = self.params.get('qgdf')
        pdf_plot = self.params.get('pdf')
        wedf = self.params.get('wedf')
        ksdf = self.params.get('ksdf')
        
        # Check smooth plotting availability
        has_smooth = (hasattr(self, 'di_points_n') and hasattr(self, 'qgdf_points') 
                    and hasattr(self, 'pdf_points') and self.di_points_n is not None
                    and self.qgdf_points is not None and self.pdf_points is not None)
        plot_smooth = plot_smooth and has_smooth
        
        # Create figure
        fig, ax1 = plt.subplots(figsize=figsize)

        # Plot QGDF if requested
        if plot in ['gdf', 'both']:
            self._plot_qgdf(ax1, x_points, qgdf_plot, plot_smooth, extra_df, wedf, ksdf)
        
        # Plot PDF if requested
        if plot in ['pdf', 'both']:
            if plot == 'pdf':
                self._plot_pdf(ax1, x_points, pdf_plot, plot_smooth, is_secondary=False)
            else:
                ax2 = ax1.twinx()
                self._plot_pdf(ax2, x_points, pdf_plot, plot_smooth, is_secondary=True)
        
        # Add bounds and formatting
        self._add_plot_formatting(ax1, plot, bounds)
        
        # Add Z0 vertical line if available
        if hasattr(self, 'z0') and self.z0 is not None:
            ax1.axvline(x=self.z0, color='magenta', linestyle='-.', linewidth=1, 
                    alpha=0.8, label=f'Z0={self.z0:.3f}')
            # Update legend to include Z0
            ax1.legend(loc='upper left', bbox_to_anchor=(0, 1))
        
        plt.tight_layout()
        plt.show()

    def _plot_qgdf(self, ax, x_points, qgdf_plot, plot_smooth, extra_df, wedf, ksdf):
        """Plot QGDF components."""
        if plot_smooth and hasattr(self, 'qgdf_points') and self.qgdf_points is not None:
            ax.plot(x_points, qgdf_plot, 'o', color='blue', label='QGDF', markersize=4)
            ax.plot(self.di_points_n, self.qgdf_points, color='blue', 
                   linestyle='-', linewidth=2, alpha=0.8)
        else:
            ax.plot(x_points, qgdf_plot, 'o-', color='blue', label='QGDF', 
                   markersize=4, linewidth=1, alpha=0.8)
        
        if extra_df:
            if wedf is not None:
                ax.plot(x_points, wedf, 's', color='lightblue', 
                       label='WEDF', markersize=3, alpha=0.8)
            if ksdf is not None:
                ax.plot(x_points, ksdf, 's', color='cyan', 
                       label='KS Points', markersize=3, alpha=0.8)
        
        ax.set_ylabel('QGDF', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax.set_ylim(0, 1)

    def _plot_pdf(self, ax, x_points, pdf_plot, plot_smooth, is_secondary=False):
        """Plot PDF components."""
        color = 'red'

        if plot_smooth and hasattr(self, 'pdf_points') and self.pdf_points is not None:
            ax.plot(x_points, pdf_plot, 'o', color=color, label='PDF', markersize=4)
            ax.plot(self.di_points_n, self.pdf_points, color=color, 
                   linestyle='-', linewidth=2, alpha=0.8)
            max_pdf = np.max(self.pdf_points)
        else:
            ax.plot(x_points, pdf_plot, 'o-', color=color, label='PDF', 
                   markersize=4, linewidth=1, alpha=0.8)
            max_pdf = np.max(pdf_plot)
        
        ax.set_ylabel('PDF', color=color)
        ax.tick_params(axis='y', labelcolor=color)
        ax.set_ylim(0, max_pdf * 1.1)
        
        if is_secondary:
            ax.legend(loc='upper right', bbox_to_anchor=(1, 1))

    def _add_plot_formatting(self, ax1, plot, bounds):
        """Add formatting, bounds, and legends to plot."""
        ax1.set_xlabel('Data Points')
        
        # Add bounds if requested
        if bounds:
            bound_info = [
                (self.params.get('DLB'), 'green', '-', 'DLB'),
                (self.params.get('DUB'), 'orange', '-', 'DUB'),
                (self.params.get('LB'), 'purple', '--', 'LB'),
                (self.params.get('UB'), 'brown', '--', 'UB')
            ]
            
            for bound, color, style, name in bound_info:
                if bound is not None:
                    ax1.axvline(x=bound, color=color, linestyle=style, linewidth=2, 
                               alpha=0.8, label=f"{name}={bound:.3f}")
            
            # Add shaded regions
            if self.params.get('LB') is not None:
                ax1.axvspan(self.data.min(), self.params['LB'], alpha=0.15, color='purple')
            if self.params.get('UB') is not None:
                ax1.axvspan(self.params['UB'], self.data.max(), alpha=0.15, color='brown')
        
        # Set limits and add grid
        data_range = self.params['DUB'] - self.params['DLB']
        padding = data_range * 0.1
        ax1.set_xlim(self.params['DLB'] - padding, self.params['DUB'] + padding)
        
        # Set title
        titles = {
            'gdf': 'QGDF' + (' with Bounds' if bounds else ''),
            'pdf': 'PDF' + (' with Bounds' if bounds else ''),
            'both': 'QGDF and PDF' + (' with Bounds' if bounds else '')
        }
        
        ax1.set_title(titles[plot])
        ax1.legend(loc='upper left', bbox_to_anchor=(0, 1))
        ax1.grid(True, alpha=0.3)
        
    
    def _get_qgdf_second_derivative(self):
        """Calculate second derivative of QGDF from stored fidelities and irrelevances."""
        if self.fi is None or self.hi is None:
            raise ValueError("Fidelities and irrelevances must be calculated before second derivative estimation.")
        
        weights = self.weights.reshape(-1, 1)
        
        # Moment calculations using QGDF's fj and hj
        f1 = np.sum(weights * self.fi, axis=0) / np.sum(weights)  # f̄Q
        h1 = np.sum(weights * self.hi, axis=0) / np.sum(weights)  # h̄Q
        f2 = np.sum(weights * self.fi**2, axis=0) / np.sum(weights)
        f3 = np.sum(weights * self.fi**3, axis=0) / np.sum(weights)
        fh = np.sum(weights * self.fi * self.hi, axis=0) / np.sum(weights)
        fh2 = np.sum(weights * self.fi * self.hi**2, axis=0) / np.sum(weights)
        f2h = np.sum(weights * self.fi**2 * self.hi, axis=0) / np.sum(weights)
        
        # QGDF uses different base equation: QGDF = (1 - h̄Q/f̄Q) / 2
        # First derivative: dQG/dZ0 = (1/SZ0) * (1 - (h̄Q)²/(f̄Q)²)
        # For second derivative, we need to differentiate this further
        
        eps = np.finfo(float).eps
        f1_safe = np.where(f1 == 0, eps, f1)
        
        # Calculate derivatives of the ratio h̄Q/f̄Q
        # d(h̄Q)/dz = -f2*S (negative of second moment scaled)
        # d(f̄Q)/dz = fh*S (cross moment scaled)
        
        dh1 = -f2 * self.S_opt
        df1 = fh * self.S_opt
        
        # For the term (h̄Q)²/(f̄Q)²
        ratio = h1 / f1_safe
        ratio_squared = ratio**2
        
        # d/dz[(h̄Q)²/(f̄Q)²] = 2*(h̄Q/f̄Q) * d/dz(h̄Q/f̄Q)
        # d/dz(h̄Q/f̄Q) = (f̄Q*dh̄Q - h̄Q*df̄Q) / (f̄Q)²
        ratio_deriv = (f1_safe * dh1 - h1 * df1) / (f1_safe**2)
        ratio_squared_deriv = 2 * ratio * ratio_deriv
        
        # The main term is (1 - (h̄Q)²/(f̄Q)²)
        # Its derivative is -ratio_squared_deriv
        main_term_deriv = -ratio_squared_deriv
        
        # Second derivative needs the derivative of main_term_deriv
        # This involves higher order moments
        d2h1 = -2 * f2h * self.S_opt  # second derivative of h̄Q
        d2f1 = (-f3 + fh2) * self.S_opt  # second derivative of f̄Q
        
        # Second derivative of the ratio using quotient rule repeatedly
        ratio_second_deriv = ((f1_safe * d2h1 - h1 * d2f1) * (f1_safe**2) - 
                                (f1_safe * dh1 - h1 * df1) * 2 * f1_safe * df1) / (f1_safe**4)
        
        ratio_squared_second_deriv = (2 * ratio_deriv**2 + 2 * ratio * ratio_second_deriv)
        
        main_term_second_deriv = -ratio_squared_second_deriv
        
        # Apply scaling factor (1/S)
        second_derivative = (1 / self.S_opt) * main_term_second_deriv
        
        return second_derivative.flatten()

    def _get_qgdf_third_derivative(self):
        """Calculate third derivative of QGDF from stored fidelities and irrelevances."""
        if self.fi is None or self.hi is None:
            raise ValueError("Fidelities and irrelevances must be calculated before third derivative estimation.")
        
        weights = self.weights.reshape(-1, 1)
        
        # All required moments for QGDF
        f1 = np.sum(weights * self.fi, axis=0) / np.sum(weights)
        h1 = np.sum(weights * self.hi, axis=0) / np.sum(weights)
        f2 = np.sum(weights * self.fi**2, axis=0) / np.sum(weights)
        f3 = np.sum(weights * self.fi**3, axis=0) / np.sum(weights)
        f4 = np.sum(weights * self.fi**4, axis=0) / np.sum(weights)
        fh = np.sum(weights * self.fi * self.hi, axis=0) / np.sum(weights)
        h2 = np.sum(weights * self.hi**2, axis=0) / np.sum(weights)
        fh2 = np.sum(weights * self.fi * self.hi**2, axis=0) / np.sum(weights)
        f2h = np.sum(weights * self.fi**2 * self.hi, axis=0) / np.sum(weights)
        f2h2 = np.sum(weights * self.fi**2 * self.hi**2, axis=0) / np.sum(weights)
        f3h = np.sum(weights * self.fi**3 * self.hi, axis=0) / np.sum(weights)
        fh3 = np.sum(weights * self.fi * self.hi**3, axis=0) / np.sum(weights)
        
        eps = np.finfo(float).eps
        f1_safe = np.where(f1 == 0, eps, f1)
        
        # Derivative calculations for QGDF
        dh1 = -f2 * self.S_opt
        df1 = fh * self.S_opt
        d2h1 = -2 * f2h * self.S_opt
        d2f1 = (-f3 + fh2) * self.S_opt
        d3h1 = -3 * f3h * self.S_opt
        d3f1 = (-f4 + 2 * f2h2) * self.S_opt
        
        # Calculate third derivative of the ratio (h̄Q/f̄Q)²
        ratio = h1 / f1_safe
        
        # First derivative of ratio
        ratio_deriv = (f1_safe * dh1 - h1 * df1) / (f1_safe**2)
        
        # Second derivative of ratio
        numerator_2nd = (f1_safe * d2h1 - h1 * d2f1) * (f1_safe**2) - (f1_safe * dh1 - h1 * df1) * 2 * f1_safe * df1
        ratio_second_deriv = numerator_2nd / (f1_safe**4)
        
        # Third derivative of ratio
        term1_3rd = f1_safe * d3h1 - h1 * d3f1
        term2_3rd = 2 * (df1 * d2h1 - dh1 * d2f1)
        term3_3rd = 6 * (f1_safe * dh1 - h1 * df1) * (df1**2) / f1_safe
        
        ratio_third_deriv = (term1_3rd * (f1_safe**2) - term2_3rd * (f1_safe**3) - term3_3rd * (f1_safe**2)) / (f1_safe**6)
        
        # Third derivative of ratio²
        ratio_squared_third_deriv = (6 * ratio_deriv * ratio_second_deriv + 2 * ratio * ratio_third_deriv)
        
        # Main term third derivative
        main_term_third_deriv = -ratio_squared_third_deriv
        
        # Apply scaling factor
        third_derivative = (1 / self.S_opt) * main_term_third_deriv
        
        return third_derivative.flatten()

    def _get_qgdf_fourth_derivative(self):
        """Calculate fourth derivative of QGDF using numerical differentiation."""
        if self.fi is None or self.hi is None:
            raise ValueError("Fidelities and irrelevances must be calculated before fourth derivative estimation.")
        
        # For fourth derivative, use numerical differentiation as it's complex
        dz = 1e-7
        
        # Get third derivatives at slightly shifted points
        zi_plus = self.zi + dz
        zi_minus = self.zi - dz
        
        # Store original zi
        original_zi = self.zi.copy()
        
        # Calculate third derivative at zi + dz
        self.zi = zi_plus
        self._calculate_fidelities_irrelevances_at_given_zi(self.zi)
        third_plus = self._get_qgdf_third_derivative()
        
        # Calculate third derivative at zi - dz  
        self.zi = zi_minus
        self._calculate_fidelities_irrelevances_at_given_zi(self.zi)
        third_minus = self._get_qgdf_third_derivative()
        
        # Restore original zi and recalculate fi, hi
        self.zi = original_zi
        self._calculate_fidelities_irrelevances_at_given_zi(self.zi)
        
        # Numerical derivative
        fourth_derivative = (third_plus - third_minus) / (2 * dz) * self.zi
        
        return fourth_derivative.flatten()

    def _calculate_fidelities_irrelevances_at_given_zi(self, zi):
        """Helper method to recalculate fidelities and irrelevances for current zi."""
        # Convert to infinite domain
        zi_n = DataConversion._convert_fininf(self.z, self.LB_opt, self.UB_opt)
        # Use given zi if provided, else use self.zi
        if zi is None:
            zi_d = self.zi
        else:
            zi_d = zi

        # Calculate R matrix
        eps = np.finfo(float).eps
        R = zi_n.reshape(-1, 1) / (zi_d + eps).reshape(1, -1)

        # Get characteristics
        gc = GnosticsCharacteristics(R=R)
        q, q1 = gc._get_q_q1(S=self.S_opt)
        
        # Store fidelities and irrelevances (using QGDF methods)
        self.fi = gc._fj(q=q, q1=q1)  # Note: using _fj for QGDF
        self.hi = gc._hj(q=q, q1=q1)  # Note: using _hj for QGDF


    def _fit_qgdf(self, plot: bool = False):
        try:
            if self.verbose:
                print("Starting QGDF fitting process...")
            
            # Step 1: Data preprocessing
            self.data = np.sort(self.data)
            self._estimate_data_bounds()
            self._transform_data_to_standard_domain()
            self._estimate_weights()
            
            # Step 2: Bounds estimation
            self._estimate_initial_probable_bounds()
            self._generate_evaluation_points()
            
            # Step 3: Get distribution function values for optimization
            self.df_values = self._get_distribution_function_values(use_wedf=self.wedf)
            
            # Step 4: Parameter optimization
            self._determine_optimization_strategy(egdf=False)  # NOTE for QGDF egdf is False
            
            # Step 5: Calculate final QGDF and PDF
            self._calculate_final_results()
            
            # Step 6: Generate smooth curves for plotting and analysis
            self._generate_smooth_curves()
            
            # Step 7: Transform bounds back to original domain
            self._transform_bounds_to_original_domain()
            
            # Mark as fitted (Step 8 is now optional via marginal_analysis())
            self._fitted = True

            # Compute Z0 point
            self._compute_z0()

            if self.verbose:
                print("QGDF fitting completed successfully.")

            if plot:
                self._plot()

            # clean up computation cache
            if self.flush:  
                self._cleanup_computation_cache()
                
        except Exception as e:
            error_msg = f"QGDF fitting failed: {e}"
            self.params['errors'].append({
                'method': '_fit_QGDF',
                'error': error_msg,
                'exception_type': type(e).__name__
            })
            
            if self.verbose:
                print(f"Error during QGDF fitting: {e}")
            raise e

    def fit(self, plot: bool = False):
        """Public method to fit QGDF and optionally plot results."""
        self._fit_qgdf(plot=plot)

    def plot(self, plot_smooth: bool = True, plot: str = 'both', bounds: bool = True, extra_df: bool = True, figsize: tuple = (12, 8)):
        """Public method to plot QGDF and/or PDF results."""
        self._plot(plot_smooth=plot_smooth, plot=plot, bounds=bounds, extra_df=extra_df, figsize=figsize)