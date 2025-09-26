"""
base class for EGDF
EGDF - Estimating Global Distribution Function.

Author: Nirmal Parmar
Machine Gnostics
"""

import numpy as np
import warnings
import logging
from machinegnostics.magcal.util.logging import get_logger
from typing import Dict, Any
from scipy.optimize import minimize
from machinegnostics.magcal.characteristics import GnosticsCharacteristics
from machinegnostics.magcal.data_conversion import DataConversion
from machinegnostics.magcal.gdf.base_distfunc import BaseDistFuncCompute
from machinegnostics.magcal.gdf.z0_estimator import Z0Estimator

class BaseEGDF(BaseDistFuncCompute):
    """
    Base class for EGDF (Estimating Global Distribution Function).
    
    This class provides a comprehensive framework for estimating global distribution
    functions with optimization capabilities and derivative analysis.
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
                         varS=False, # NOTE for EGDfF varS is always False 
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

        # logger
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if verbose else logging.WARNING)
        self.logger.debug(f"{self.__class__.__name__} initialized:")
        

    def _compute_egdf_core(self, S, LB, UB, zi_data=None, zi_eval=None):
        """Core EGDF computation with caching."""
        # self.logger.info("Starting core EGDF computation.")
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
        gc = GnosticsCharacteristics(R=R, verbose=self.verbose)
        q, q1 = gc._get_q_q1(S=S)
        
        # Calculate fidelities and irrelevances
        fi = gc._fi(q=q, q1=q1)
        hi = gc._hi(q=q, q1=q1)
        
        # Estimate EGDF
        return self._estimate_egdf_from_moments(fi, hi), fi, hi

    def _estimate_egdf_from_moments(self, fidelities, irrelevances):
        """Estimate EGDF from fidelities and irrelevances."""
        # self.logger.info("Estimating EGDF from moments.")
        weights = self._computation_cache['weights_normalized'].reshape(-1, 1)
        
        mean_fidelity = np.sum(weights * fidelities, axis=0) / np.sum(weights)
        mean_irrelevance = np.sum(weights * irrelevances, axis=0) / np.sum(weights)
        
        M_zi = np.sqrt(mean_fidelity**2 + mean_irrelevance**2)
        M_zi = np.where(M_zi == 0, self._NUMERICAL_EPS, M_zi)
        
        egdf_values = (1 - mean_irrelevance / M_zi) / 2
        egdf_values = np.maximum.accumulate(egdf_values)
        egdf_values = np.clip(egdf_values, 0, 1)
        
        return egdf_values.flatten()

    # NOTE: PDF calculation as mentioned in a new book
    # def _calculate_pdf_from_moments(self, fidelities, irrelevances):
    #     """Calculate PDF from fidelities and irrelevances."""
    #     weights = self._computation_cache['weights_normalized'].reshape(-1, 1)
        
    #     mean_fidelity = np.sum(weights * fidelities, axis=0) / np.sum(weights)
    #     mean_irrelevance = np.sum(weights * irrelevances, axis=0) / np.sum(weights)
        
    #     F2 = np.sum(weights * fidelities**2, axis=0) / np.sum(weights)
    #     FH = np.sum(weights * fidelities * irrelevances, axis=0) / np.sum(weights)
        
    #     M_zi = np.sqrt(mean_fidelity**2 + mean_irrelevance**2)
    #     M_zi = np.where(M_zi == 0, self._NUMERICAL_EPS, M_zi)
    #     M_zi_cubed = M_zi**3
        
    #     numerator = (mean_fidelity**2) * F2 + mean_fidelity * mean_irrelevance * FH
    #     S_value = self.S_opt if hasattr(self, 'S_opt') else 1.0
    #     density = (1 / S_value) * (numerator / M_zi_cubed)
        
    #     if np.any(density < 0):
    #         warnings.warn("PDF contains negative values, indicating potential non-homogeneous data", RuntimeWarning)
    #     return density.flatten()

    def _calculate_pdf_from_moments(self, fidelities, irrelevances): # PDF
        """Calculate first derivative of EGDF (which is the PDF) from stored fidelities and irrelevances."""
        self.logger.info("Calculating PDF from moments.")
        if fidelities is None or irrelevances is None:
            raise ValueError("Fidelities and irrelevances must be calculated before first derivative estimation.")
        
        weights = self.weights.reshape(-1, 1)
        
        # First order moments
        f1 = np.sum(weights * fidelities, axis=0) / np.sum(weights)  # mean_fidelity
        h1 = np.sum(weights * irrelevances, axis=0) / np.sum(weights)  # mean_irrelevance

        # Second order moments (scaled by S as in MATLAB)
        f2s = np.sum(weights * (fidelities**2 / self.S_opt), axis=0) / np.sum(weights)
        fhs = np.sum(weights * (fidelities * irrelevances / self.S_opt), axis=0) / np.sum(weights)
        
        # Calculate denominator w = (f1^2 + h1^2)^(3/2)
        w = (f1**2 + h1**2)**(3/2)
        eps = np.finfo(float).eps
        w = np.where(w == 0, eps, w)
        
        # First derivative formula from MATLAB: y = (f1^2 * f2s + f1 * h1 * fhs) / w
        numerator = f1**2 * f2s + f1 * h1 * fhs
        first_derivative = numerator / w
        # first_derivative = first_derivative / self.zi
        
        # if np.any(first_derivative < 0):
        #     warnings.warn("EGDF first derivative (PDF) contains negative values, indicating potential non-homogeneous data", RuntimeWarning)
        return first_derivative.flatten()


    def _calculate_final_results(self):
        """Calculate final EGDF and PDF with optimized parameters."""
        self.logger.info("Calculating final EGDF and PDF with optimized parameters.")
        # Convert to infinite domain
        # zi_n = DataConversion._convert_fininf(self.z, self.LB_opt, self.UB_opt)
        zi_d = DataConversion._convert_fininf(self.z, self.LB_opt, self.UB_opt)
        self.zi = zi_d
        
        # Calculate EGDF and get moments
        egdf_values, fi, hi = self._compute_egdf_core(self.S_opt, self.LB_opt, self.UB_opt)
        
        # Store for derivative calculations
        self.fi = fi
        self.hi = hi
        self.egdf = egdf_values
        self.pdf = self._calculate_pdf_from_moments(fi, hi)
        
        if self.catch:
            self.logger.info("Catching parameters for later use.")
            self.params.update({
                'egdf': self.egdf.copy(),
                'pdf': self.pdf.copy(),
                'zi': self.zi.copy()
            })

    def _generate_smooth_curves(self):
        """Generate smooth curves for plotting and analysis."""
        self.logger.info("Generating smooth curves for EGDF and PDF.")
        try:
            # Generate smooth EGDF and PDF
            smooth_egdf, self.smooth_fi, self.smooth_hi = self._compute_egdf_core(
                self.S_opt, self.LB_opt, self.UB_opt,
                zi_data=self.z_points_n, zi_eval=self.z
            )
            
            smooth_pdf = self._calculate_pdf_from_moments(self.smooth_fi, self.smooth_hi)
            
            self.egdf_points = smooth_egdf
            self.pdf_points = smooth_pdf
            
            # Store zi_n for derivative calculations
            self.zi_n = DataConversion._convert_fininf(self.z_points_n, self.LB_opt, self.UB_opt)
            
            # Mark as generated
            self._computation_cache['smooth_curves_generated'] = True
            
            if self.catch:
                self.logger.info("Catching parameters for later use.")
                self.params.update({
                    'egdf_points': self.egdf_points.copy(),
                    'pdf_points': self.pdf_points.copy(),
                    'zi_points': self.zi_n.copy()
                })

            self.logger.info(f"Generated smooth curves with {self.n_points} points.")

        except Exception as e:
            # Log the error
            error_msg = f"Could not generate smooth curves: {e}"
            self.logger.error(error_msg)
            self.params['errors'].append({
                'method': '_generate_smooth_curves',
                'error': error_msg,
                'exception_type': type(e).__name__
            })
            self.logger.warning(f"Could not generate smooth curves: {e}")
            # Create fallback points using original data
            self.egdf_points = self.egdf.copy() if hasattr(self, 'egdf') else None
            self.pdf_points = self.pdf.copy() if hasattr(self, 'pdf') else None
            self._computation_cache['smooth_curves_generated'] = False

    
    def _plot(self, plot_smooth: bool = True, plot: str = 'both', bounds: bool = True, extra_df: bool = True, figsize: tuple = (12, 8)):
        """Enhanced plotting with better organization."""
        self.logger.info("Starting plot generation.")
        
        import matplotlib.pyplot as plt

        if plot_smooth and (len(self.data) > self.max_data_size) and self.verbose:
            self.logger.warning(f"Given data size ({len(self.data)}) exceeds max_data_size ({self.max_data_size}). For optimal compute performance, set 'plot_smooth=False', or 'max_data_size' to a larger value whichever is appropriate.")

        if not self.catch:
            self.logger.warning("Plot is not available with argument catch=False")
            return
        
        if not self._fitted:
            self.logger.error("Must fit EGDF before plotting.")
            raise RuntimeError("Must fit EGDF before plotting.")
        
        # Validate plot parameter
        if plot not in ['gdf', 'pdf', 'both']:
            self.logger.error("Invalid plot parameter. Must be 'gdf', 'pdf', or 'both'.")
            raise ValueError("plot parameter must be 'gdf', 'pdf', or 'both'")
        
        # Check data availability
        if plot in ['gdf', 'both'] and self.params.get('egdf') is None:
            self.logger.error("EGDF must be calculated before plotting GDF")
            raise ValueError("EGDF must be calculated before plotting GDF")
        if plot in ['pdf', 'both'] and self.params.get('pdf') is None:
            self.logger.error("PDF must be calculated before plotting PDF")
            raise ValueError("PDF must be calculated before plotting PDF")
        
        # Prepare data
        self.logger.info("Preparing data for plotting.")    
        x_points = self.data
        egdf_plot = self.params.get('egdf')
        pdf_plot = self.params.get('pdf')
        wedf = self.params.get('wedf')
        ksdf = self.params.get('ksdf')
        
        # Check smooth plotting availability
        has_smooth = (hasattr(self, 'di_points_n') and hasattr(self, 'egdf_points') 
                    and hasattr(self, 'pdf_points') and self.di_points_n is not None
                    and self.egdf_points is not None and self.pdf_points is not None)
        plot_smooth = plot_smooth and has_smooth
        
        # Create figure
        fig, ax1 = plt.subplots(figsize=figsize)

        # Plot EGDF if requested
        if plot in ['gdf', 'both']:
            self._plot_egdf(ax1, x_points, egdf_plot, plot_smooth, extra_df, wedf, ksdf)
        
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

    def _plot_egdf(self, ax, x_points, egdf_plot, plot_smooth, extra_df, wedf, ksdf):
        """Plot EGDF components."""
        self.logger.info("Plotting EGDF.")
        if plot_smooth and hasattr(self, 'egdf_points') and self.egdf_points is not None:
            ax.plot(x_points, egdf_plot, 'o', color='blue', label='EGDF', markersize=4)
            ax.plot(self.di_points_n, self.egdf_points, color='blue', 
                   linestyle='-', linewidth=2, alpha=0.8)
        else:
            ax.plot(x_points, egdf_plot, 'o-', color='blue', label='EGDF', 
                   markersize=4, linewidth=1, alpha=0.8)
        
        if extra_df:
            if wedf is not None:
                ax.plot(x_points, wedf, 's', color='lightblue', 
                       label='WEDF', markersize=3, alpha=0.8)
            if ksdf is not None:
                ax.plot(x_points, ksdf, 's', color='cyan', 
                       label='KS Points', markersize=3, alpha=0.8)
        
        ax.set_ylabel('EGDF', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax.set_ylim(0, 1)

    def _plot_pdf(self, ax, x_points, pdf_plot, plot_smooth, is_secondary=False):
        """Plot PDF components."""
        self.logger.info("Plotting PDF.")
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
            'gdf': 'EGDF' + (' with Bounds' if bounds else ''),
            'pdf': 'PDF' + (' with Bounds' if bounds else ''),
            'both': 'EGDF and PDF' + (' with Bounds' if bounds else '')
        }
        
        ax1.set_title(titles[plot])
        ax1.legend(loc='upper left', bbox_to_anchor=(0, 1))
        ax1.grid(True, alpha=0.3)



    # =============================================================================
    # Derivative
    # =============================================================================
    def _get_egdf_second_derivative(self):
        """Calculate second derivative of EGDF from stored fidelities and irrelevances."""
        self.logger.info("Calculating second derivative of EGDF.")
        if self.fi is None or self.hi is None:
            self.logger.error("Fidelities and irrelevances must be calculated before second derivative estimation.")
            raise ValueError("Fidelities and irrelevances must be calculated before second derivative estimation.")
        
        weights = self.weights.reshape(-1, 1)
        
        # Moment calculations
        f1 = np.sum(weights * self.fi, axis=0) / np.sum(weights)
        h1 = np.sum(weights * self.hi, axis=0) / np.sum(weights)
        f2 = np.sum(weights * self.fi**2, axis=0) / np.sum(weights)
        f3 = np.sum(weights * self.fi**3, axis=0) / np.sum(weights)
        fh = np.sum(weights * self.fi * self.hi, axis=0) / np.sum(weights)
        fh2 = np.sum(weights * self.fi * self.hi**2, axis=0) / np.sum(weights)
        f2h = np.sum(weights * self.fi**2 * self.hi, axis=0) / np.sum(weights)
        
        # Calculate components
        b = f1**2 * f2 + f1 * h1 * fh
        d = f1**2 + h1**2
        eps = np.finfo(float).eps
        d = np.where(d == 0, eps, d)
        
        # Following
        term1 = f1 * (h1 * (f3 - fh2) - f2 * fh)
        term2 = 2 * f1**2 * f2h + h1 * fh**2
        term3 = (6 * b * (f1 * fh - h1 * f2)) / d
        
        d2 = -1 / (d**(1.5)) * (2 * (term1 - term2) + term3)
        second_derivative = d2 / (self.S_opt**2)
        # second_derivative = second_derivative / self.zi**2 
        self.logger.info("Second derivative calculation completed.")
        return second_derivative.flatten()

    def _get_egdf_third_derivative(self):
        """Calculate third derivative of EGDF from stored fidelities and irrelevances."""
        self.logger.info("Calculating third derivative of EGDF.")
        if self.fi is None or self.hi is None:
            self.logger.error("Fidelities and irrelevances must be calculated before third derivative estimation.")
            raise ValueError("Fidelities and irrelevances must be calculated before third derivative estimation.")
        
        weights = self.weights.reshape(-1, 1)
        
        # All required moments
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
        
        # Following
        # Derivative calculations
        dh1 = -f2
        df1 = fh
        df2 = 2 * f2h
        dfh = -f3 + fh2
        dfh2 = -2 * f3h + fh3
        df3 = 3 * f3h
        df2h = -f4 + 2 * f2h2
        
        # u4 and its derivative
        u4 = h1 * f3 - h1 * fh2 - f2 * fh
        du4 = dh1 * f3 + h1 * df3 - dh1 * fh2 - h1 * dfh2 - df2 * fh - f2 * dfh
        
        # u and its derivative
        u = f1 * u4
        du = df1 * u4 + f1 * du4
        
        # v components
        v4a = (f1**2) * f2h
        dv4a = 2 * f1 * df1 * f2h + (f1**2) * df2h
        v4b = h1 * fh**2
        dv4b = dh1 * (fh**2) + 2 * h1 * fh * dfh
        
        v = 2 * v4a + v4b
        dv = 2 * dv4a + dv4b
        
        # x components
        x4a = f1**2 * f2 + f1 * h1 * fh
        dx4a = 2 * f1 * df1 * f2 + (f1**2) * df2 + df1 * h1 * fh + f1 * dh1 * fh + f1 * h1 * dfh
        x4b = f1 * fh - h1 * f2
        dx4b = df1 * fh + f1 * dfh - dh1 * f2 - h1 * df2
        
        x = 6 * x4a * x4b
        dx = 6 * (dx4a * x4b + x4a * dx4b)
        
        # d components
        d = f1**2 + h1**2
        dd = 2 * (f1 * df1 + h1 * dh1)
        eps = np.finfo(float).eps
        d = np.where(d == 0, eps, d)
        
        # Final calculation
        term1 = (du - dv) / (d**1.5) - (1.5 * (u - v)) / (d**2.5) * dd
        term2 = dx / (d**2.5) - (2.5 * x) / (d**3.5) * dd
        
        d3p = -2 * term1 - term2
        third_derivative = 2 * d3p / (self.S_opt**3)
        # third_derivative = third_derivative / (self.zi**3)
        self.logger.info("Third derivative calculation completed.")
        return third_derivative.flatten()

    def _get_egdf_fourth_derivative(self):
        """Calculate fourth derivative of EGDF using numerical differentiation."""
        self.logger.info("Calculating fourth derivative of EGDF using numerical differentiation.")
        if self.fi is None or self.hi is None:
            self.logger.error("Fidelities and irrelevances must be calculated before fourth derivative estimation.")
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
        third_plus = self._get_egdf_third_derivative()
        
        # Calculate third derivative at zi - dz  
        self.zi = zi_minus
        self._calculate_fidelities_irrelevances_at_given_zi(self.zi)
        third_minus = self._get_egdf_third_derivative()
        
        # Restore original zi and recalculate fi, hi
        self.zi = original_zi
        self._calculate_fidelities_irrelevances_at_given_zi(self.zi)
        
        # Numerical derivative
        fourth_derivative = (third_plus - third_minus) / (2 * dz) * self.zi

        self.logger.info("Fourth derivative calculation completed.")
        return fourth_derivative.flatten()

    def _calculate_fidelities_irrelevances_at_given_zi(self, zi):
        """Helper method to recalculate fidelities and irrelevances for current zi."""
        self.logger.info("Recalculating fidelities and irrelevances for given zi.")
        if self.LB_opt is None or self.UB_opt is None or self.S_opt is None:
            self.logger.error("Optimized parameters LB_opt, UB_opt, and S_opt must be set before recalculating fidelities and irrelevances.")
        # Convert to infinite domain
        zi_n = DataConversion._convert_fininf(self.z, self.LB_opt, self.UB_opt)
        # is zi given then use it, else use self.zi
        if zi is None:
            zi_d = self.zi
        else:
            zi_d = zi

        # Calculate R matrix
        eps = np.finfo(float).eps
        R = zi_n.reshape(-1, 1) / (zi_d + eps).reshape(1, -1)

        # Get characteristics
        gc = GnosticsCharacteristics(R=R, verbose=self.verbose)
        q, q1 = gc._get_q_q1(S=self.S_opt)
        
        # Store fidelities and irrelevances
        self.fi = gc._fi(q=q, q1=q1)
        self.hi = gc._hi(q=q, q1=q1)

    def _get_results(self)-> dict:
        """Return fitting results."""
        self.logger.info("Retrieving fitting results.")
        if not self._fitted:
            raise RuntimeError("Must fit EGDF before getting results.")
        
        # selected key from params if exists
        keys = ['DLB', 'DUB', 'LB', 'UB', 'S_opt', 'z0', 'egdf', 'pdf', 
                'egdf_points', 'pdf_points', 'zi', 'zi_points', 'weights']
        results = {key: self.params.get(key) for key in keys if key in self.params}
        return results

    # =============================================================================
    # MAIN FITTING PROCESS
    # =============================================================================
    
    def _fit_egdf(self, plot:bool = True):
        """Main fitting process with improved organization."""
        self.logger.info("Starting EGDF fitting process.")
        try:
            # Step 1: Data preprocessing
            self.logger.info("Starting data preprocessing.")
            self.data = np.sort(self.data)
            self._estimate_data_bounds()
            self._transform_data_to_standard_domain()
            self._estimate_weights()
            
            # Step 2: Bounds estimation
            self.logger.info("Starting bounds estimation.")
            self._estimate_initial_probable_bounds()
            self._generate_evaluation_points()
            
            # Step 3: Get distribution function values for optimization
            self.logger.info("Getting distribution function values for optimization.")
            self.df_values = self._get_distribution_function_values(use_wedf=self.wedf)
            
            # Step 4: Parameter optimization
            self.logger.info("Starting parameter optimization.")
            self._determine_optimization_strategy()
            
            # Step 5: Calculate final EGDF and PDF
            self.logger.info("Calculating final EGDF and PDF.")
            self._calculate_final_results()
            
            # Step 6: Generate smooth curves for plotting and analysis
            self.logger.info("Generating smooth curves for plotting and analysis.")
            self._generate_smooth_curves()
            
            # Step 7: Transform bounds back to original domain
            self.logger.info("Transforming bounds back to original domain.")
            self._transform_bounds_to_original_domain()
            
            # Mark as fitted (Step 8 is now optional via marginal_analysis())
            self._fitted = True

            # Compute Z0 point
            self.logger.info("Computing Z0 point.")
            self._compute_z0()

            self.logger.info("EGDF fitting completed successfully.")

            if plot:
                self.logger.info("Plotting results.")
                self._plot()

            # clean up computation cache
            if self.flush:
                self.logger.info("Cleaning up computation cache.")
                self._cleanup_computation_cache()
                
        except Exception as e:
            error_msg = f"EGDF fitting failed: {e}"
            self.logger.error(error_msg)
            self.params['errors'].append({
                'method': '_fit_egdf',
                'error': error_msg,
                'exception_type': type(e).__name__
            })
            self.logger.info(f"Error during EGDF fitting: {e}")
            raise e
        
    # z0 compute
    def _compute_z0(self, optimize: bool = None):
        """
        Compute the Z0 point where PDF is maximum using the Z0Estimator class.
        
        Parameters:
        -----------
        optimize : bool, optional
            If True, use interpolation-based methods for higher accuracy.
            If False, use simple linear search on existing points.
            If None, uses the instance's z0_optimize setting.
        """
        self.logger.info("Starting Z0 computation.")
        if self.z is None:
            self.logger.error("Data must be transformed (self.z) before Z0 estimation.")
            raise ValueError("Data must be transformed (self.z) before Z0 estimation.")
        
        # Use provided optimize parameter or fall back to instance setting
        use_optimize = optimize if optimize is not None else self.z0_optimize

        self.logger.info("EGDF: Computing Z0 point using Z0Estimator...")

        try:
            # Create Z0Estimator instance with proper constructor signature
            z0_estimator = Z0Estimator(
                gdf_object=self,  # Pass the EGDF object itself
                optimize=use_optimize,
                verbose=self.verbose
            )
            
            # Call fit() method to estimate Z0
            self.z0 = z0_estimator.fit()
            
            # Get estimation info for debugging and storage
            if self.catch:
                estimation_info = z0_estimator.get_estimation_info()
                self.params.update({
                    'z0': float(self.z0) if self.z0 is not None else None,
                    'z0_method': estimation_info.get('z0_method', 'unknown'),
                    'z0_estimation_info': estimation_info
                })
            
            method_used = z0_estimator.get_estimation_info().get('z0_method', 'unknown')
            self.logger.info(f'EGDF: Z0 point computed successfully, (method: {method_used})')

        except Exception as e:
            # Log the error
            error_msg = f"Z0 estimation failed: {str(e)}"
            self.logger.error(error_msg)
            self.params['errors'].append({
                'method': '_compute_z0',
                'error': error_msg,
                'exception_type': type(e).__name__
            })

            self.logger.warning(f"Warning: Z0Estimator failed with error: {e}")
            self.logger.info("Falling back to simple maximum finding...")

            # Fallback to simple maximum finding
            self.logger.info("Using fallback method for Z0 computation.")
            self._compute_z0_fallback()
            
            if self.catch:
                self.logger.info("Catching fallback Z0 parameters for later use.")
                self.params.update({
                    'z0': float(self.z0),
                    'z0_method': 'fallback_simple_maximum',
                    'z0_estimation_info': {'error': str(e)}
                })

    def _compute_z0_fallback(self):
        """
        Fallback method for Z0 computation using simple maximum finding.
        """
        self.logger.info("Starting fallback Z0 computation.")

        if not hasattr(self, 'di_points_n') or not hasattr(self, 'pdf_points'):
            self.logger.error("Both 'di_points_n' and 'pdf_points' must be defined for Z0 computation.")
            raise ValueError("Both 'di_points_n' and 'pdf_points' must be defined for Z0 computation.")
        
        self.logger.info('Using fallback method for Z0 point...')
        
        # Find index with maximum PDF
        max_idx = np.argmax(self.pdf_points)
        self.z0 = self.di_points_n[max_idx]

        self.logger.info(f"Z0 point (fallback method).")

    def analyze_z0(self, figsize: tuple = (12, 6)) -> Dict[str, Any]:
        """
        Analyze and visualize Z0 estimation results.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size for the plot
            
        Returns:
        --------
        Dict[str, Any]
            Z0 analysis information
        """
        self.logger.info("Starting Z0 analysis.")
        if not hasattr(self, 'z0') or self.z0 is None:
            self.logger.error("Z0 must be computed before analysis. Call fit() first.")
            raise ValueError("Z0 must be computed before analysis. Call fit() first.")
        
        # Create Z0Estimator for analysis
        z0_estimator = Z0Estimator(
            gdf_object=self,
            optimize=self.z0_optimize,
            verbose=self.verbose
        )
        
        # Re-estimate for analysis (this is safe since it's already computed)
        z0_estimator.fit()
        
        # Get detailed info
        analysis_info = z0_estimator.get_estimation_info()
        
        # Create visualization
        z0_estimator.plot_z0_analysis(figsize=figsize)
        
        self.logger.info("Z0 analysis completed.")
        return analysis_info
    