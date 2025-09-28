'''
base QLDF class
Quantifying Local Distribution Functions

Author: Nirmal Parmar
Machine Gnostics
'''

import numpy as np
import warnings
from scipy.optimize import minimize
from typing import Dict, Any
import logging
from machinegnostics.magcal.util.logging import get_logger
from machinegnostics.magcal.characteristics import GnosticsCharacteristics
from machinegnostics.magcal.data_conversion import DataConversion
from machinegnostics.magcal.gdf.base_qgdf import BaseQGDF
from machinegnostics.magcal.scale_param import ScaleParam
from machinegnostics.magcal.gdf.z0_estimator import Z0Estimator

class BaseQLDF(BaseQGDF):
    '''Base QLDF class'''
    def __init__(self,
                 data: np.ndarray,
                 DLB: float = None,
                 DUB: float = None,
                 LB: float = None,
                 UB: float = None,
                 S = 'auto',
                 varS: bool = False,
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
        self.varS = varS # QLDF specific
        self.z0_optimize = z0_optimize # QLDF specific
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
        self._fitted = False  # To track if fit has been called

        # Store initial parameters if catching
        if self.catch:
            self._store_initial_params()

        # Validate all inputs
        # self._validate_inputs()

        # logger setup
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if verbose else logging.WARNING)
        self.logger.debug(f"{self.__class__.__name__} initialized:")

        # if S is float or int and is greater than 2, warn user
        if (isinstance(self.S, float) or isinstance(self.S, int)) and self.S > 2:
            self.logger.warning("S is greater than 2, which may not suitable for quantifying local distribution estimation. Consider using in range [0, 2]")
            warnings.warn("S is greater than 2, which may not suitable for quantifying local distribution estimation. Consider using in range [0, 2]", UserWarning)

        

    def _fit_qldf(self, plot: bool = True):
        """Fit the QLDF model to the data."""
        self.logger.debug("Starting QLDF fitting process...")
        try:
            # Step 1: Data preprocessing
            self.logger.info("Preprocessing data...")
            self.data = np.sort(self.data)
            self._estimate_data_bounds()
            self._transform_data_to_standard_domain()
            self._estimate_weights()
            
            # Step 2: Bounds estimation
            self.logger.info("Estimating initial probable bounds...")
            self._estimate_initial_probable_bounds()
            self._generate_evaluation_points()
            
            # Step 3: Get distribution function values for optimization
            self.logger.info("Calculating distribution function values...")
            self.df_values = self._get_distribution_function_values(use_wedf=self.wedf)
            
            # Step 4: Parameter optimization
            self.logger.info("Optimizing parameters...")
            self._determine_optimization_strategy(egdf=False)  # QLDF does not use egdf 
            
            # Step 5: Calculate final QLDF and PDF
            self.logger.info("Computing final QLDF and PDF...")
            self._compute_final_results()
            
            # Step 6: Generate smooth curves for plotting and analysis
            self.logger.info("Generating smooth curves for analysis...")
            self._generate_smooth_curves()
            
            # Step 7: Transform bounds back to original domain
            self.logger.info("Transforming bounds back to original domain...")
            self._transform_bounds_to_original_domain()
            # Mark as fitted (Step 8 is now optional via marginal_analysis())
            self._fitted = True

            # Step 8: Z0 estimate with Z0Estimator
            self.logger.info("Estimating Z0...")    
            self._compute_z0(optimize=self.z0_optimize) 
            # derivatives
            # self._calculate_all_derivatives()
            
            # # Step 9: varS       
            if self.varS:
                self.logger.info("Calculating variable S parameter...")
                self._varS_calculation()
                self.logger.info("Recomputing final results with variable S...")
                self._compute_final_results_varS()
                self.logger.info("Generating smooth curves with variable S...")
                self._generate_smooth_curves_varS()

            # Step 10: Z0 re-estimate with varS if enabled
            if self.varS:
                self.logger.info("Re-estimating Z0 with variable S...")
                self._compute_z0(optimize=self.z0_optimize) 

            self.logger.info("QLDF fitting completed successfully.")

            if plot:
                self.logger.info("Plotting results...")
                self._plot()
            
            # clean up computation cache
            if self.flush:  
                self.logger.info("Flushing computation cache to free memory...")
                self._cleanup_computation_cache()
                    
        except Exception as e:
            # log error
            error_msg = f"QLDF fitting failed: {e}"
            self.logger.error(error_msg)
            self.params['errors'].append({
                'method': '_fit_qldf',
                'error': error_msg,
                'exception_type': type(e).__name__
            })
            self.logger.info(f"Error during QLDF fitting: {e}")
            raise e


    def _compute_qldf_core(self, S, LB, UB, zi_data=None, zi_eval=None):
        """Core computation for the QLDF model."""
        self.logger.debug("Computing core QLDF values...")

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
        
        # Calculate quantifying fidelities and irrelevances
        fj = gc._fj(q=q, q1=q1)  # quantifying fidelities
        hj = gc._hj(q=q, q1=q1)  # quantifying irrelevances
        return self._estimate_qldf_from_moments(fj, hj), fj, hj

    def _estimate_qldf_from_moments(self, fidelity, irrelevance):
        """Estimate the QLDF from moments using equation (15.33): QLDF = (1 - h_QL)/2."""
        self.logger.debug("Estimating QLDF from moments...")

        weights = self.weights.reshape(-1, 1)

        # Calculate weighted mean of quantifying irrelevances (h_QL)
        mean_irrelevance = np.sum(weights * irrelevance, axis=0) / np.sum(weights)
        # hQL
        hQL = mean_irrelevance / (np.sqrt(1 + mean_irrelevance**2) + self._NUMERICAL_EPS)
        # Apply equation (15.33): QLDF = (1 - h_QL)/2
        qldf_values = (1 - hQL) / 2

        return qldf_values.flatten()

    def _compute_final_results(self):
        """Compute the final results for the QLDF model."""
        self.logger.debug("Computing final QLDF results...")

        # Convert data to infinite domain
        zi_d = DataConversion._convert_fininf(self.z, self.LB_opt, self.UB_opt)
        self.zi = zi_d

        # Calculate QLDF and get moments
        qldf_values, fj, hj = self._compute_qldf_core(self.S_opt, self.LB_opt, self.UB_opt)

        # Store for derivative calculations
        self.fj = fj  # quantifying fidelities
        self.hj = hj  # quantifying irrelevances

        # # varS - Variable S parameter
        # if self.varS:
        #     fj_m = np.sum(self.fj * self.weights, axis=0) / np.sum(self.weights)
        #     scale = ScaleParam()
        #     self.S_var = np.abs(scale._gscale_loc(fj_m) * self.S_opt) # NOTE fi or fj?
        #     # cap value for minimum S_var array
        #     self.S_var = np.maximum(self.S_var, 0.1)
        #     qldf_values, fj, hj = self._compute_qldf_core(self.S_var, self.LB_opt, self.UB_opt)
        #     self.fj = fj
        #     self.hj = hj

        self.qldf = qldf_values
        self.pdf = self._compute_qldf_pdf(self.fj, self.hj)
        
        if self.catch:
            self.params.update({
                'qldf': self.qldf.copy(),
                'pdf': self.pdf.copy(),
                'zi': self.zi.copy(),
                # 'S_var': self.S_var.copy() if self.varS else None
            })

    def _compute_qldf_pdf(self, fj, hj):
        """Compute the PDF for the QLDF model using equation (15.34): dQL/dZ₀ = (1/SZ₀) * f̄Q/((1 + (h̄Q)²)^(3/2))."""
        self.logger.debug("Computing PDF for QLDF...")

        weights = self.weights.reshape(-1, 1)

        # Calculate weighted means of quantifying fidelities and irrelevances
        fQ_mean = np.sum(weights * fj, axis=0) / np.sum(weights)  # f̄Q
        hQ_mean = np.sum(weights * hj, axis=0) / np.sum(weights)  # h̄Q

        # hQL
        hQL = hQ_mean / (np.sqrt(1 + hQ_mean**2) + self._NUMERICAL_EPS)

        # Apply equation (15.34): dQL/dZ₀ = (1/SZ₀) * f̄Q/((1 + (h̄Q)²)^(3/2))
        # Note: We use S instead of SZ₀ for the scaling factor
        denominator = (1 + hQL**2)**(3/2)
        
        # Handle division by zero
        eps = np.finfo(float).eps
        denominator = np.where(denominator == 0, eps, denominator)

        pdf_values = (1 / self.S_opt) * fQ_mean / denominator

        return pdf_values.flatten()

    def _generate_smooth_curves(self):
        """Generate smooth curves for plotting and analysis - QLDF."""
        self.logger.debug("Generating smooth curves for QLDF...")

        try:
            if self.verbose and not self.varS:
                self.logger.info("Generating smooth curves without varying S...")

            smooth_qldf, self.smooth_fj, self.smooth_hj = self._compute_qldf_core(
                self.S_opt, self.LB_opt, self.UB_opt,
                zi_data=self.z_points_n, zi_eval=self.z
            )
            smooth_pdf = self._compute_qldf_pdf(self.smooth_fj, self.smooth_hj) 
    
            self.qldf_points = smooth_qldf
            self.pdf_points = smooth_pdf
            
            # Store zi_n for derivative calculations
            self.zi_n = DataConversion._convert_fininf(self.z_points_n, self.LB_opt, self.UB_opt)
            
            # Mark as generated
            self._computation_cache['smooth_curves_generated'] = True
            
            if self.catch:
                self.params.update({
                    'qldf_points': self.qldf_points.copy(),
                    'pdf_points': self.pdf_points.copy(),
                    'zi_points': self.zi_n.copy()
                })
            
            if self.verbose and not self.varS:
                self.logger.info(f"Generated smooth curves with {self.n_points} points.")
                
        except Exception as e:
            # log error
            error_msg = f"Smooth curve generation failed: {e}"
            self.logger.error(error_msg)
            self.params['errors'].append({
                'method': '_generate_smooth_curves',
                'error': error_msg,
                'exception_type': type(e).__name__
            })

            self.logger.warning(f"Warning: Could not generate smooth curves: {e}")

            # Create fallback points using original data
            self.qldf_points = self.qldf.copy() if hasattr(self, 'qldf') else None
            self.pdf_points = self.pdf.copy() if hasattr(self, 'pdf') else None
            self._computation_cache['smooth_curves_generated'] = False

    
    def _plot(self, plot_smooth: bool = True, plot: str = 'both', bounds: bool = True, extra_df: bool = True, figsize: tuple = (12, 8)):
        """Enhanced plotting with better organization."""
        self.logger.info("Plotting QLDF results...")

        import matplotlib.pyplot as plt
    
        if plot_smooth and (len(self.data) > self.max_data_size) and self.verbose:
            self.logger.info(f"Warning: Given data size ({len(self.data)}) exceeds max_data_size ({self.max_data_size}). For optimal compute performance, set 'plot_smooth=False', or 'max_data_size' to a larger value whichever is appropriate.")
    
        if not self.catch:
            self.logger.info("Plot is not available with argument catch=False")
            return
        
        if not self._fitted:
            self.logger.info("QLDF is not fitted yet.")
            raise RuntimeError("Must fit QLDF before plotting.")
        
        # Validate plot parameter
        if plot not in ['gdf', 'pdf', 'both']:
            self.logger.error("Invalid plot parameter. Must be 'gdf', 'pdf', or 'both'.")
            raise ValueError("plot parameter must be 'gdf', 'pdf', or 'both'")
        
        # Check data availability
        if plot in ['gdf', 'both'] and self.params.get('qldf') is None:
            self.logger.error("QLDF must be calculated before plotting GDF")
            raise ValueError("QLDF must be calculated before plotting GDF")
        if plot in ['pdf', 'both'] and self.params.get('pdf') is None:
            self.logger.error("PDF must be calculated before plotting PDF")
            raise ValueError("PDF must be calculated before plotting PDF")
        
        # Prepare data
        x_points = self.data
        qldf_plot = self.params.get('qldf')
        pdf_plot = self.params.get('pdf')
        wedf = self.params.get('wedf')
        ksdf = self.params.get('ksdf')
        
        # Check smooth plotting availability
        has_smooth = (hasattr(self, 'z_points_n') and hasattr(self, 'qldf_points') 
                     and hasattr(self, 'pdf_points') and self.z_points_n is not None
                     and self.qldf_points is not None and self.pdf_points is not None)
        plot_smooth = plot_smooth and has_smooth
        
        # Create figure
        fig, ax1 = plt.subplots(figsize=figsize)
    
        # Plot QLDF if requested
        if plot in ['gdf', 'both']:
            self._plot_qldf(ax1, x_points, qldf_plot, plot_smooth, extra_df, wedf, ksdf)
        
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
    
    def _plot_pdf(self, ax, x_points, pdf_plot, plot_smooth, is_secondary=False):
        """Plot PDF components."""
        self.logger.debug("Plotting PDF...")
        import numpy as np  # Add numpy import
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

    def _plot_qldf(self, ax, x_points, qldf_plot, plot_smooth, extra_df, wedf, ksdf):
        """Plot QLDF components."""
        self.logger.debug("Plotting QLDF...")
        if plot_smooth and hasattr(self, 'qldf_points') and self.qldf_points is not None:
            ax.plot(x_points, qldf_plot, 'o', color='blue', label='QLDF', markersize=4)
            ax.plot(self.di_points_n, self.qldf_points, color='blue', 
                linestyle='-', linewidth=2, alpha=0.8)
        else:
            ax.plot(x_points, qldf_plot, 'o-', color='blue', label='QLDF', 
                markersize=4, linewidth=1, alpha=0.8)
        
        if extra_df:
            if wedf is not None:
                ax.plot(x_points, wedf, 's', color='lightblue', 
                    label='WEDF', markersize=3, alpha=0.8)
            if ksdf is not None:
                ax.plot(x_points, ksdf, 's', color='cyan', 
                    label='KS Points', markersize=3, alpha=0.8)
        
        ax.set_ylabel('QLDF', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax.set_ylim(0, 1)

    def _add_plot_formatting(self, ax1, plot, bounds):
        """Add formatting, bounds, and legends to plot."""
        self.logger.debug("Adding plot formatting...")
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
            'gdf': 'QLDF' + (' with Bounds' if bounds else ''),
            'pdf': 'PDF' + (' with Bounds' if bounds else ''),
            'both': 'QLDF and PDF' + (' with Bounds' if bounds else '')
        }
        
        ax1.set_title(titles[plot])
        ax1.legend(loc='upper left', bbox_to_anchor=(0, 1))
        ax1.grid(True, alpha=0.3)

    def _get_qldf_second_derivative(self, fj=None, hj=None):
        """
        Calculate second derivative of QLDF using mathematical derivation.
        
        Starting from: dQL/dZ₀ = (1/S) * f̄Q/((1 + (h̄Q)²)^(3/2))
        
        Second derivative: d²QL/dZ₀² = (1/S) * d/dZ₀[f̄Q/((1 + (h̄Q)²)^(3/2))]
        """
        self.logger.debug("Calculating second derivative of QLDF...")
        if fj is None or hj is None:
            fj = self.fj
            hj = self.hj

        if fj is None or hj is None:
            self.logger.error("Quantifying fidelities and irrelevances must be calculated before second derivative estimation.")
            raise ValueError("Quantifying fidelities and irrelevances must be calculated before second derivative estimation.")
        
        weights = self.weights.reshape(-1, 1)
        
        # Calculate weighted means and their derivatives
        fQ_mean = np.sum(weights * fj, axis=0) / np.sum(weights)  # f̄Q
        hQ_mean = np.sum(weights * hj, axis=0) / np.sum(weights)  # h̄Q
        
        # For derivatives, we need: d(f̄Q)/dZ₀ and d(h̄Q)/dZ₀
        # These are approximated by the variance-like terms
        dfQ_dz = np.sum(weights * (fj - fQ_mean) * self.zi.reshape(-1, 1), axis=0) / np.sum(weights)
        dhQ_dz = np.sum(weights * (hj - hQ_mean) * self.zi.reshape(-1, 1), axis=0) / np.sum(weights)
        
        # Apply quotient rule: d/dx[u/v] = (v*du - u*dv)/v²
        # where u = f̄Q and v = (1 + (h̄Q)²)^(3/2)
        
        u = fQ_mean
        v = (1 + hQ_mean**2)**(3/2)
        du_dz = dfQ_dz
        dv_dz = (3/2) * (1 + hQ_mean**2)**(1/2) * 2 * hQ_mean * dhQ_dz
        
        # Second derivative using quotient rule
        second_derivative = (1 / self.S_opt) * (v * du_dz - u * dv_dz) / (v**2)
        
        return second_derivative.flatten()

    def _get_qldf_third_derivative(self, fj=None, hj=None):
        """
        Calculate third derivative of QLDF using mathematical derivation.
        
        This involves differentiating the second derivative expression.
        """
        self.logger.debug("Calculating third derivative of QLDF...")

        if fj is None or hj is None:
            fj = self.fj
            hj = self.hj

        if fj is None or hj is None:
            self.logger.error("Quantifying fidelities and irrelevances must be calculated before third derivative estimation.")
            raise ValueError("Quantifying fidelities and irrelevances must be calculated before third derivative estimation.")
        
        weights = self.weights.reshape(-1, 1)
        
        # Calculate weighted means and derivatives
        fQ_mean = np.sum(weights * fj, axis=0) / np.sum(weights)
        hQ_mean = np.sum(weights * hj, axis=0) / np.sum(weights)
        
        # First derivatives
        dfQ_dz = np.sum(weights * (fj - fQ_mean) * self.zi.reshape(-1, 1), axis=0) / np.sum(weights)
        dhQ_dz = np.sum(weights * (hj - hQ_mean) * self.zi.reshape(-1, 1), axis=0) / np.sum(weights)
        
        # Second derivatives (approximated)
        d2fQ_dz2 = np.sum(weights * (fj - fQ_mean) * (self.zi.reshape(-1, 1))**2, axis=0) / np.sum(weights)
        d2hQ_dz2 = np.sum(weights * (hj - hQ_mean) * (self.zi.reshape(-1, 1))**2, axis=0) / np.sum(weights)
        
        # Complex expression for third derivative - this is a simplified approximation
        # Full derivation would be extremely complex
        
        # Terms for the third derivative calculation
        term1 = d2fQ_dz2 / (1 + hQ_mean**2)**(3/2)
        term2 = -3 * dfQ_dz * hQ_mean * dhQ_dz / (1 + hQ_mean**2)**(5/2)
        term3 = -3 * fQ_mean * d2hQ_dz2 * hQ_mean / (1 + hQ_mean**2)**(5/2)
        term4 = 15 * fQ_mean * hQ_mean**2 * (dhQ_dz)**2 / (1 + hQ_mean**2)**(7/2)
        
        third_derivative = (1 / self.S_opt) * (term1 + term2 + term3 + term4)
        
        return third_derivative.flatten()
    
    def _get_qldf_fourth_derivative(self, fj=None, hj=None):
        """Calculate fourth derivative of QLDF using numerical differentiation."""
        self.logger.debug("Calculating fourth derivative of QLDF...")

        if fj is None or hj is None:
            fj = self.fj
            hj = self.hj

        if fj is None or hj is None:
            self.logger.error("Quantifying fidelities and irrelevances must be calculated before fourth derivative estimation.")
            raise ValueError("Quantifying fidelities and irrelevances must be calculated before fourth derivative estimation.")
        
        # For fourth derivative, use numerical differentiation as it's extremely complex
        dz = 1e-7
        
        # Get third derivatives at slightly shifted points
        zi_plus = self.zi + dz
        zi_minus = self.zi - dz
        
        # Store original zi
        original_zi = self.zi.copy()
        
        # Calculate third derivative at zi + dz
        self.zi = zi_plus
        self._calculate_fidelities_irrelevances_at_given_zi(self.zi)
        third_plus = self._get_qldf_third_derivative()
        
        # Calculate third derivative at zi - dz  
        self.zi = zi_minus
        self._calculate_fidelities_irrelevances_at_given_zi(self.zi)
        third_minus = self._get_qldf_third_derivative()
        
        # Restore original zi and recalculate fj, hj
        self.zi = original_zi
        self._calculate_fidelities_irrelevances_at_given_zi(self.zi)
        
        # Numerical derivative
        fourth_derivative = (third_plus - third_minus) / (2 * dz) * self.zi
        
        return fourth_derivative.flatten()
    
    def _get_qldf_derivatives_numerical(self, order=2, h=1e-6):
        """
        Calculate QLDF derivatives using numerical differentiation.
        This is more reliable for higher-order derivatives.
        
        Parameters:
        -----------
        order : int
            Order of derivative (2, 3, or 4)
        h : float
            Step size for numerical differentiation
        """
        self.logger.debug(f"Calculating {order}th derivative of QLDF using numerical differentiation...")
        if not hasattr(self, 'pdf_points') or self.pdf_points is None:
            self.logger.error("PDF must be calculated before derivative estimation.")
            raise ValueError("PDF must be calculated before derivative estimation.")
        
        from scipy.misc import derivative
        
        # Create interpolation function for PDF
        from scipy.interpolate import interp1d
        pdf_interp = interp1d(self.di_points_n, self.pdf_points, 
                            kind='cubic', bounds_error=False, fill_value=0)
        
        # Calculate derivatives at data points
        derivatives = []
        for z_val in self.data:
            if order == 2:
                deriv = derivative(pdf_interp, z_val, dx=h, n=1, order=3)
            elif order == 3:
                deriv = derivative(pdf_interp, z_val, dx=h, n=2, order=5)
            elif order == 4:
                deriv = derivative(pdf_interp, z_val, dx=h, n=3, order=7)
            else:
                raise ValueError("Order must be 2, 3, or 4")
            
            derivatives.append(deriv)
        
        return np.array(derivatives)

    def _calculate_fidelities_irrelevances_at_given_zi(self, zi):
        """Helper method to recalculate quantifying fidelities and irrelevances for current zi."""
        self.logger.debug("Recalculating quantifying fidelities and irrelevances for given zi...")  

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
        gc = GnosticsCharacteristics(R=R, verbose=self.verbose)
        q, q1 = gc._get_q_q1(S=self.S_opt)
        
        # Store quantifying fidelities and irrelevances
        self.fj = gc._fj(q=q, q1=q1)  # quantifying fidelities
        self.hj = gc._hj(q=q, q1=q1)  # quantifying irrelevances
    
    def _get_results(self)-> dict:
        """Return fitting results."""
        self.logger.debug("Retrieving QLDF results...")

        if not self._fitted:
            self.logger.error("QLDF must be fitted before getting results.")
            raise RuntimeError("Must fit QLDF before getting results.")
        
        # selected key from params if exists
        keys = ['DLB', 'DUB', 'LB', 'UB', 'S_opt', 'z0', 'qldf', 'pdf',
                'qldf_points', 'pdf_points', 'zi', 'zi_points', 'weights']
        results = {key: self.params.get(key) for key in keys if key in self.params}
        return results
    
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
        self.logger.debug("Computing Z0 point...")
        if self.z is None:
            self.logger.error("Data must be transformed (self.z) before Z0 estimation.")
            raise ValueError("Data must be transformed (self.z) before Z0 estimation.")
        
        # Use provided optimize parameter or fall back to instance setting
        use_optimize = optimize if optimize is not None else self.z0_optimize
        
        self.logger.info('QLDF: Computing Z0 point using Z0Estimator...')

        try:
            # Create Z0Estimator instance with proper constructor signature
            z0_estimator = Z0Estimator(
                gdf_object=self,  # Pass the QLDF object itself
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
            self.logger.info(f'QLDF: Z0 point computed successfully, (method: {method_used})')

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
            self._compute_z0_fallback()
            
            if self.catch:
                self.params.update({
                    'z0': float(self.z0),
                    'z0_method': 'fallback_simple_maximum',
                    'z0_estimation_info': {'error': str(e)}
                })

    def _compute_z0_fallback(self):
        """
        Fallback method for Z0 computation using simple maximum finding.
        """
        self.logger.info("Computing Z0 point using fallback method (simple maximum finding)...")

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
        self.logger.debug("Analyzing Z0 estimation...")

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
        
        return analysis_info
    
    def _calculate_all_derivatives(self):
        """Calculate all derivatives and store in params."""
        self.logger.debug("Calculating all QLDF derivatives...")

        if not self._fitted:
            self.logger.error("QLDF must be fitted before calculating derivatives.")
            raise RuntimeError("Must fit QLDF before calculating derivatives.")
        
        try:
            # Calculate derivatives using analytical methods
            second_deriv = self._get_qldf_second_derivative()
            third_deriv = self._get_qldf_third_derivative()
            fourth_deriv = self._get_qldf_fourth_derivative()
            
            # Store in params
            if self.catch:
                self.params.update({
                    'second_derivative': second_deriv.copy(),
                    'third_derivative': third_deriv.copy(),
                    'fourth_derivative': fourth_deriv.copy()
                })

            self.logger.info("QLDF derivatives calculated and stored successfully.")

        except Exception as e:
            # Log error
            error_msg = f"Derivative calculation failed: {e}"
            self.logger.error(error_msg)
            self.params['errors'].append({
                'method': '_calculate_all_derivatives',
                'error': error_msg,
                'exception_type': type(e).__name__
            })

            self.logger.error(f"Warning: Could not calculate derivatives: {e}")

            # Fallback to numerical differentiation
            try:
                second_deriv_num = self._get_qldf_derivatives_numerical(order=2)
                third_deriv_num = self._get_qldf_derivatives_numerical(order=3)
                fourth_deriv_num = self._get_qldf_derivatives_numerical(order=4)
                
                if self.catch:
                    self.params.update({
                        'second_derivative': second_deriv_num.copy(),
                        'third_derivative': third_deriv_num.copy(),
                        'fourth_derivative': fourth_deriv_num.copy()
                    })

                self.logger.info("QLDF derivatives calculated using numerical differentiation and stored successfully.")

            except Exception as ne:
                # Log numerical differentiation error
                num_error_msg = f"Numerical derivative calculation failed: {ne}"
                self.logger.error(num_error_msg)
                self.params['errors'].append({
                    'method': '_calculate_all_derivatives_numerical',
                    'error': num_error_msg,
                    'exception_type': type(ne).__name__
                })

                self.logger.warning(f"Warning: Could not calculate numerical derivatives: {ne}")

    def _estimate_s0_sigma(self, z0, s_local, s_global, mode="sum"):
        """
        Estimate S0 and sigma given z0 (float), s_local (array), and s_global (float).
        
        Parameters
        ----------
        z0 : float
            Mean value of the data.
        s_local : array-like
            Local scale parameters.
        s_global : float
            Global scale parameter.
        mode : str
            "mean" -> match average predicted to s_global
            "sum"  -> match sum predicted to s_global
        
        Returns
        -------
        S0, sigma : floats
            Estimated parameters.
        """
        self.logger.info("Estimating S0 and sigma...")
        s_local = np.asarray(s_local)

        def objective(params):
            S0, sigma = params
            preds = S0 * np.exp(sigma * z0) * s_local
            if mode == "mean":
                target = preds.mean()
            elif mode == "sum":
                target = preds.sum()
            else:
                raise ValueError("mode must be 'mean' or 'sum'")
            return (s_global - target) ** 2

        # Initial guess
        p0 = [1.0, 0.0]

        res = minimize(objective, p0, method="Nelder-Mead")
        return res.x[0], res.x[1]
    
    def _varS_calculation(self):
        """Calculate varS if enabled."""
        self.logger.debug("Calculating varS for QLDF...")

        from machinegnostics import variance

        self.logger.info("Calculating varS for QLDF...")
        # estimate fi hi at z0
        gc, q, q1 = self._calculate_gcq_at_given_zi(self.z0)

        fi_z0 = gc._fj(q=q, q1=q1)

        scale = ScaleParam()
        self.S_local = scale._gscale_loc(fi_z0)

        self.S_local = np.maximum(self.S_local, 0.1)  # cap value for minimum S_local array

        # # s0 # NOTE for future exploration
        # self.S0, self.sigma = self._estimate_s0_sigma(
        #     z0=self.z0,
        #     s_local=fi_z0,
        #     s_global=self.S_opt,
        #     mode="sum"
        # )

        # Svar
        self.S_var = self.S_local * self.S_opt
        return self.S_var

    def _compute_final_results_varS(self):
        """Compute the final results for the QLDF model."""
        self.logger.info("Computing final results for QLDF with varS...")
        # Implement final results computation logic here
        # zi_d = DataConversion._convert_fininf(self.z, self.LB_opt, self.UB_opt)
        # self.zi = zi_d

        qldf_values, fj, hj = self._compute_qldf_core(self.S_var, self.LB_opt, self.UB_opt)
        self.fj = fj
        self.hj = hj

        self.qldf = qldf_values
        self.pdf = self._compute_qldf_pdf(self.fj, self.hj)
        
        if self.catch:
            self.params.update({
                'qldf': self.qldf.copy(),
                'pdf': self.pdf.copy(),
                'zi': self.zi.copy(),
                'S_var': self.S_var.copy() if self.varS else None
            })

    def _generate_smooth_curves_varS(self):
        """Generate smooth curves for plotting and analysis - QLDF."""
        self.logger.info("Generating smooth curves for QLDF with varS...")
        try:
            self.logger.info("Generating smooth curves with varying S...")

            smooth_qldf, self.smooth_fj, self.smooth_hj = self._compute_qldf_core(
                self.S_var, self.LB_opt, self.UB_opt,
                zi_data=self.z_points_n, zi_eval=self.z
            )
            smooth_pdf = self._compute_qldf_pdf(self.smooth_fj, self.smooth_hj) 
        
            self.qldf_points = smooth_qldf
            self.pdf_points = smooth_pdf
            
            # Mark as generated
            self._computation_cache['smooth_curves_generated'] = True
            
            if self.catch:
                self.params.update({
                    'qldf_points': self.qldf_points.copy(),
                    'pdf_points': self.pdf_points.copy(),
                    'zi_points': self.zi_n.copy()
                })
           
            self.logger.info(f"Generated smooth curves with {self.n_points} points.")

        except Exception as e:
            # log error
            error_msg = f"Smooth curve generation failed: {e}"
            self.logger.error(error_msg)
            self.params['errors'].append({
                'method': '_generate_smooth_curves_varS',
                'error': error_msg,
                'exception_type': type(e).__name__
            })

            self.logger.warning(f"Warning: Could not generate smooth curves: {e}")

            # Create fallback points using original data
            self.qldf_points = self.qldf.copy() if hasattr(self, 'qldf') else None
            self.pdf_points = self.pdf.copy() if hasattr(self, 'pdf') else None
            self._computation_cache['smooth_curves_generated'] = False