'''
QGDF: Quantifying Global Distribution Functions

Author: Nirmal Parmar
Machine Gnostics
'''
import numpy as np
import warnings
import logging
from machinegnostics.magcal.util.logging import get_logger
from typing import Dict, Any
from machinegnostics.magcal.gdf.base_distfunc import BaseDistFuncCompute
from machinegnostics.magcal.data_conversion import DataConversion
from machinegnostics.magcal.characteristics import GnosticsCharacteristics
from machinegnostics.magcal.gdf.z0_estimator import Z0Estimator

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
                 tolerance: float = 1e-6,
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

        # logger setup
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if verbose else logging.WARNING)
        self.logger.debug(f"{self.__class__.__name__} initialized:")

    def _compute_qgdf_core(self, S, LB, UB, zi_data=None, zi_eval=None):
        """Core QGDF computation with caching."""
        self.logger.info("Computing QGDF core.")
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
        fj = gc._fj(q=q, q1=q1)
        hj = gc._hj(q=q, q1=q1)
        
        # Estimate QGDF
        return self._estimate_qgdf_from_moments(fj, hj), fj, hj

    
    def _estimate_qgdf_from_moments(self, fidelities, irrelevances):
        """Main QGDF estimation with robust numerical stabilization."""
        self.logger.info("Estimating QGDF from moments (robust).")
        try:
            if fidelities is None or irrelevances is None:
                raise ValueError("fidelities and irrelevances must be provided")
    
            # Choose high-precision accumulator when available
            acc_dtype = np.float128 if hasattr(np, "float128") else np.float64
            eps = getattr(self, "_NUMERICAL_EPS", np.finfo(float).eps)
    
            # Weights: use cached normalized if present, else normalize here
            w = self._computation_cache.get("weights_normalized", None)
            if w is None:
                self.logger.warning("Weights not normalized in cache; normalizing locally.")
                w = self.weights
            w = np.asarray(w, dtype=acc_dtype).reshape(-1, 1)
    
            wsum = np.sum(w, dtype=acc_dtype)
            if not np.isfinite(wsum) or wsum <= 0:
                self.logger.warning("Invalid weights; falling back to uniform weights.")
                n = fidelities.shape[0]
                w = np.full((n, 1), 1.0 / max(n, 1), dtype=acc_dtype)
            else:
                w = w / wsum  # ensure sum(w) == 1 in high precision
    
            # Per-column scaling (same scale for f and h, preserves ratio)
            # scale_j = max(max_i |f_ij|, max_i |h_ij|, 1)
            abs_f_max = np.max(np.abs(fidelities), axis=0)
            abs_h_max = np.max(np.abs(irrelevances), axis=0)
            scale = np.maximum(np.maximum(abs_f_max, abs_h_max), 1.0).astype(acc_dtype)
    
            f_s = (fidelities.astype(acc_dtype) / scale)
            h_s = (irrelevances.astype(acc_dtype) / scale)
    
            # Weighted means in high precision (weights already sum to 1)
            mean_f = np.sum(w * f_s, axis=0, dtype=acc_dtype)
            mean_h = np.sum(w * h_s, axis=0, dtype=acc_dtype)
    
            # Stable denominator via hypot (sqrt(f^2 + h^2))
            denom = np.hypot(mean_f, mean_h).astype(acc_dtype)
            denom = np.where(denom > 0, denom, eps)
    
            # Ratio is scale-invariant due to common scaling
            ratio = mean_h / denom
    
            # Clamp tiny numerical excursions
            ratio = np.clip(ratio, -1.0 + 1e-15, 1.0 - 1e-15)
    
            qgdf_values = (1.0 - ratio) * 0.5
    
            # Clean non-finite values
            qgdf_values = np.nan_to_num(qgdf_values, nan=0.0, posinf=1.0, neginf=0.0)
    
            # Enforce monotonicity and bounds
            qgdf_values = np.maximum.accumulate(qgdf_values)
            qgdf_values = np.clip(qgdf_values, 0.0, 1.0)
    
            return np.asarray(qgdf_values, dtype=float).flatten()
    
        except Exception as e:
            error_msg = f"Exception in QGDF estimation: {e}"
            self.logger.error(error_msg)
            self.params['errors'].append({
                'method': '_estimate_qgdf_from_moments',
                'error': error_msg,
                'exception_type': type(e).__name__
            })
            # Best-effort safe fallback
            return np.zeros(fidelities.shape[1], dtype=float)
    
    def _calculate_pdf_from_moments(self, fidelities, irrelevances):
        self.logger.info("Calculating PDF from moments.")
        """Calculate PDF using definition: dQG/dZ0 = (1/(S*Z0)) * [1 - (h̄Q/f̄Q)^2]."""
        if fidelities is None or irrelevances is None:
            self.logger.error("Fidelities and irrelevances must be calculated first.")
            raise ValueError("Fidelities and irrelevances must be calculated first")

        # Weights
        weights = self._computation_cache['weights_normalized']
        if weights is None:
            self.logger.warning("Weights not normalized in cache; falling back to self.weights.")
            weights = self.weights
        weights = weights.reshape(-1, 1)

        # Numeric guards
        eps = np.finfo(float).eps
        min_safe = np.sqrt(eps)

        # Weighted means
        mean_fidelity = np.sum(weights * fidelities, axis=0) / np.sum(weights)
        mean_irrelevance = np.sum(weights * irrelevances, axis=0) / np.sum(weights)

        # Guard small fidelity
        mean_fidelity_safe = np.where(np.abs(mean_fidelity) < min_safe,
                                      np.sign(mean_fidelity) * min_safe,
                                      mean_fidelity)
        ratio = mean_irrelevance / mean_fidelity_safe
        # Optional tight clamp for stability without distorting theory
        ratio = np.clip(ratio, -1 + 1e-12, 1 - 1e-12)

        # Select Z0 vector based on context
        Z0_used = None
        try:
            # If computed over evaluation grid (smooth curves)
            if hasattr(self, 'z_points_n') and fidelities.shape[1] == len(self.z_points_n):
                Z0_used = self.z_points_n
            # Else if computed directly over transformed data
            elif hasattr(self, 'z') and fidelities.shape[1] == len(self.z):
                Z0_used = self.z
        except Exception:
            Z0_used = None

        # Fallback to scalar z0 if available; else ones
        if Z0_used is None:
            if hasattr(self, 'z0') and self.z0 is not None:
                Z0_used = np.full_like(ratio, fill_value=float(self.z0), dtype=float)
            else:
                Z0_used = np.ones_like(ratio, dtype=float)

        # Guard small Z0
        Z0_used_safe = np.where(np.abs(Z0_used) < min_safe,
                                np.sign(Z0_used) * min_safe,
                                Z0_used)

        S_value = self.S_opt if hasattr(self, 'S_opt') else 1.0
        denom = S_value * Z0_used_safe
        denom = np.where(np.abs(denom) < min_safe, np.sign(denom) * min_safe, denom)

        pdf_values = (1.0 / denom) * (1.0 - ratio**2)
        # Ensure finite values
        pdf_values = np.where(np.isfinite(pdf_values), pdf_values, 0.0)
        return pdf_values.flatten()

    def _calculate_final_results(self):
        """Calculate final QGDF and PDF with optimized parameters."""
        self.logger.info("Calculating final QGDF and PDF results.")
        # Convert to infinite domain
        # zi_n = DataConversion._convert_fininf(self.z, self.LB_opt, self.UB_opt)
        zi_d = DataConversion._convert_fininf(self.z, self.LB_opt, self.UB_opt)
        self.zi = zi_d

        # Calculate QGDF and get moments
        qgdf_values, fj, hj = self._compute_qgdf_core(self.S_opt, self.LB_opt, self.UB_opt)

        # Store for derivative calculations
        self.fj = fj
        self.hj = hj
        self.qgdf = qgdf_values
        self.pdf = self._calculate_pdf_from_moments(fj, hj)
        
        if self.catch:
            self.params.update({
                'qgdf': self.qgdf.copy(),
                'pdf': self.pdf.copy(),
                'zi': self.zi.copy()
            })

    def _generate_smooth_curves(self):
        """Generate smooth curves for plotting and analysis."""
        self.logger.info("Generating smooth curves for QGDF and PDF.")
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
            self.qgdf_points = self.qgdf.copy() if hasattr(self, 'qgdf') else None
            self.pdf_points = self.pdf.copy() if hasattr(self, 'pdf') else None
            self._computation_cache['smooth_curves_generated'] = False

    def _get_results(self)-> dict:
        """Return fitting results."""
        self.logger.info("Getting results from QGDF fitting.")

        if not self._fitted:
            error_msg = "Must fit QGDF before getting results."
            self.logger.error(error_msg)
            self.params['errors'].append({
                'method': '_get_results',
                'error': error_msg,
                'exception_type': 'RuntimeError'
            })
            raise RuntimeError("Must fit QGDF before getting results.")

        # selected key from params if exists
        keys = ['DLB', 'DUB', 'LB', 'UB', 'S_opt', 'z0', 'qgdf', 'pdf', 
                'qgdf_points', 'pdf_points', 'zi', 'zi_points', 'weights']
        results = {key: self.params.get(key) for key in keys if key in self.params}
        return results


    def _plot(self, plot_smooth: bool = True, plot: str = 'both', bounds: bool = True, extra_df: bool = True, figsize: tuple = (12, 8)):
        """Enhanced plotting with better organization."""
        self.logger.info("Plotting QGDF and PDF results.")
        import matplotlib.pyplot as plt

        if plot_smooth and (len(self.data) > self.max_data_size) and self.verbose:
            self.logger.warning(f"Given data size ({len(self.data)}) exceeds max_data_size ({self.max_data_size}). For optimal compute performance, set 'plot_smooth=False', or 'max_data_size' to a larger value whichever is appropriate.")

        if not self.catch:
            self.logger.warning("Plot is not available with argument catch=False")
            return
        
        if not self._fitted:
            self.logger.error("Must fit QGDF before plotting.")
            raise RuntimeError("Must fit QGDF before plotting.")
        
        # Validate plot parameter
        if plot not in ['gdf', 'pdf', 'both']:
            self.logger.error("Invalid plot parameter.")
            raise ValueError("plot parameter must be 'gdf', 'pdf', or 'both'")
        
        # Check data availability
        if plot in ['gdf', 'both'] and self.params.get('qgdf') is None:
            self.logger.error("QGDF must be calculated before plotting GDF.")
            raise ValueError("QGDF must be calculated before plotting GDF")
        if plot in ['pdf', 'both'] and self.params.get('pdf') is None:
            self.logger.error("PDF must be calculated before plotting PDF.")
            raise ValueError("PDF must be calculated before plotting PDF.")

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
        self.logger.info("Plotting QGDF components.")
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
        self.logger.info("Plotting PDF components.")
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
        self.logger.info("Adding plot formatting and bounds.")
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
        """Calculate second derivative of QGDF with corrected mathematical formulation."""
        self.logger.info("Calculating second derivative of QGDF.")
        if self.fj is None or self.hj is None:
            self.logger.error("Fidelities and irrelevances must be calculated before second derivative estimation.")
            raise ValueError("Fidelities and irrelevances must be calculated before second derivative estimation.")
        
        weights = self.weights.reshape(-1, 1)
        
        # Calculate all required moments
        f1 = np.sum(weights * self.fj, axis=0) / np.sum(weights)  # f̄Q
        h1 = np.sum(weights * self.hj, axis=0) / np.sum(weights)  # h̄Q
        f2 = np.sum(weights * self.fj**2, axis=0) / np.sum(weights)
        h2 = np.sum(weights * self.hj**2, axis=0) / np.sum(weights)
        fh = np.sum(weights * self.fj * self.hj, axis=0) / np.sum(weights)
        
        # Additional moments for second derivative
        f3 = np.sum(weights * self.fj**3, axis=0) / np.sum(weights)
        h3 = np.sum(weights * self.hj**3, axis=0) / np.sum(weights)
        f2h = np.sum(weights * self.fj**2 * self.hj, axis=0) / np.sum(weights)
        fh2 = np.sum(weights * self.fj * self.hj**2, axis=0) / np.sum(weights)
        
        eps = np.finfo(float).eps
        f1_safe = np.where(np.abs(f1) < eps, np.sign(f1) * eps, f1)
        
        # CORRECTED: Based on the actual QGDF equation QGDF = (1 + h_GQ)/2
        # where h_GQ = h_zj / √(1 + h_zj²) and h_zj = h̄Q / √(f̄Q² - h̄Q²)
        
        # Calculate first derivatives of weighted means
        # These are derived from the variance-covariance relationships
        df1_dz = (f2 - f1**2) / self.S_opt  # Corrected: variance formula
        dh1_dz = (h2 - h1**2) / self.S_opt  # Corrected: variance formula
        
        # Calculate second derivatives
        d2f1_dz2 = (f3 - 3*f1*f2 + 2*f1**3) / (self.S_opt**2)  # Third central moment
        d2h1_dz2 = (h3 - 3*h1*h2 + 2*h1**3) / (self.S_opt**2)  # Third central moment
        
        # Calculate derivatives of h_zj = h̄Q / √(f̄Q² - h̄Q²)
        denominator_squared = f1_safe**2 - h1**2
        denominator_squared = np.maximum(denominator_squared, eps)
        denominator = np.sqrt(denominator_squared)
        
        h_zj = h1 / denominator
        
        # First derivative of h_zj using quotient rule
        d_numerator = dh1_dz
        d_denominator = (f1_safe * df1_dz - h1 * dh1_dz) / denominator
        
        dh_zj_dz = (d_numerator * denominator - h_zj * d_denominator) / denominator
        
        # Second derivative of h_zj (more complex)
        d2_numerator = d2h1_dz2
        # For d²(denominator), we need more careful calculation
        temp_term = f1_safe * d2f1_dz2 - h1 * d2h1_dz2 - df1_dz**2 - dh1_dz**2
        d2_denominator = (temp_term * denominator - d_denominator**2) / denominator
        
        d2h_zj_dz2 = ((d2_numerator * denominator - d_numerator * d_denominator) * denominator - 
                       (d_numerator * denominator - h_zj * d_denominator) * d_denominator) / (denominator**2)
        
        # Calculate derivatives of h_GQ = h_zj / √(1 + h_zj²)
        h_zj_squared = np.minimum(h_zj**2, 1e10)  # Prevent overflow
        h_gq_denominator = np.sqrt(1 + h_zj_squared)
        
        # First derivative of h_GQ
        dh_gq_dz = dh_zj_dz / (h_gq_denominator**3)
        
        # Second derivative of h_GQ
        term1 = d2h_zj_dz2 / (h_gq_denominator**3)
        term2 = -3 * dh_zj_dz**2 * h_zj / (h_gq_denominator**5)
        
        d2h_gq_dz2 = term1 + term2
        
        # Finally, second derivative of QGDF = (1/2) * d²(h_GQ)/dz²
        second_derivative = 0.5 * d2h_gq_dz2
        
        return second_derivative.flatten()

    def _get_qgdf_third_derivative(self):
        """Calculate third derivative of QGDF with corrected mathematical formulation."""
        self.logger.info("Calculating third derivative of QGDF.")
        if self.fj is None or self.hj is None:
            self.logger.error("Fidelities and irrelevances must be calculated before third derivative estimation.")
            raise ValueError("Fidelities and irrelevances must be calculated before third derivative estimation.")
        
        weights = self.weights.reshape(-1, 1)
        
        # Calculate all required moments up to 4th order
        f1 = np.sum(weights * self.fj, axis=0) / np.sum(weights)
        h1 = np.sum(weights * self.hj, axis=0) / np.sum(weights)
        f2 = np.sum(weights * self.fj**2, axis=0) / np.sum(weights)
        h2 = np.sum(weights * self.hj**2, axis=0) / np.sum(weights)
        f3 = np.sum(weights * self.fj**3, axis=0) / np.sum(weights)
        h3 = np.sum(weights * self.hj**3, axis=0) / np.sum(weights)
        f4 = np.sum(weights * self.fj**4, axis=0) / np.sum(weights)
        h4 = np.sum(weights * self.hj**4, axis=0) / np.sum(weights)
        
        eps = np.finfo(float).eps
        f1_safe = np.where(np.abs(f1) < eps, np.sign(f1) * eps, f1)
        
        # Calculate derivatives up to third order
        df1_dz = (f2 - f1**2) / self.S_opt
        dh1_dz = (h2 - h1**2) / self.S_opt
        
        d2f1_dz2 = (f3 - 3*f1*f2 + 2*f1**3) / (self.S_opt**2)
        d2h1_dz2 = (h3 - 3*h1*h2 + 2*h1**3) / (self.S_opt**2)
        
        d3f1_dz3 = (f4 - 4*f1*f3 + 6*f1**2*f2 - 3*f1**4) / (self.S_opt**3)
        d3h1_dz3 = (h4 - 4*h1*h3 + 6*h1**2*h2 - 3*h1**4) / (self.S_opt**3)
        
        # Calculate h_zj and its derivatives (simplified approach)
        denominator_squared = f1_safe**2 - h1**2
        denominator_squared = np.maximum(denominator_squared, eps)
        denominator = np.sqrt(denominator_squared)
        
        h_zj = h1 / denominator
        
        # For third derivative, use numerical differentiation as analytical form is extremely complex
        h = 1e-6 * np.std(self.data) if np.std(self.data) > 0 else 1e-6
        
        # Store original values
        original_zi = self.zi.copy()
        original_fi = self.fj.copy()
        original_hi = self.hj.copy()
        
        try:
            # Calculate second derivative at nearby points
            second_derivs = []
            points = [-h, 0, h]
            
            for delta in points:
                self.zi = original_zi + delta
                self._calculate_fidelities_irrelevances_at_given_zi(self.zi)
                second_deriv = self._get_qgdf_second_derivative()
                second_derivs.append(second_deriv)
            
            # Use finite difference formula for third derivative
            # f'''(x) ≈ [f''(x+h) - f''(x-h)] / (2h)
            third_derivative = (second_derivs[2] - second_derivs[0]) / (2 * h)
            
            return third_derivative.flatten()
            
        finally:
            # Always restore original state
            self.zi = original_zi
            self.fj = original_fi
            self.hj = original_hi

    def _get_qgdf_fourth_derivative(self):
        """Calculate fourth derivative of QGDF using corrected numerical differentiation."""
        self.logger.info("Calculating fourth derivative of QGDF.")
        if self.fj is None or self.hj is None:
            self.logger.error("Fidelities and irrelevances must be calculated before fourth derivative estimation.")
            raise ValueError("Fidelities and irrelevances must be calculated before fourth derivative estimation.")
        
        # Use adaptive step size based on data scale
        data_scale = np.std(self.data) if np.std(self.data) > 0 else 1.0
        h = max(1e-6 * data_scale, 1e-10)
        
        # Store original state
        original_fi = self.fj.copy()
        original_hi = self.hj.copy()
        original_zi = self.zi.copy()
        
        try:
            # Use 5-point stencil for better accuracy
            # f''''(x) ≈ [f'''(x-2h) - 8f'''(x-h) + 8f'''(x+h) - f'''(x+2h)] / (12h)
            points = [-2*h, -h, 0, h, 2*h]
            third_derivatives = []
            
            for delta in points:
                self.zi = original_zi + delta
                self._calculate_fidelities_irrelevances_at_given_zi(self.zi)
                third_deriv = self._get_qgdf_third_derivative()
                third_derivatives.append(third_deriv)
            
            # Apply 5-point finite difference formula
            fourth_derivative = (third_derivatives[0] - 8*third_derivatives[1] + 
                                8*third_derivatives[3] - third_derivatives[4]) / (12*h)
            
            # REMOVED THE INCORRECT MULTIPLICATION BY self.zi
            # The original code incorrectly multiplied by self.zi
            
            return fourth_derivative.flatten()
            
        finally:
            # Always restore original state
            self.fj = original_fi
            self.hj = original_hi  
            self.zi = original_zi

    def _calculate_fidelities_irrelevances_at_given_zi_corrected(self, zi):
        """Helper method to recalculate fidelities and irrelevances for current zi."""
        self.logger.info("Calculating fidelities and irrelevances at given zi.")
        # FIXED: Convert the data points to infinite domain, not the evaluation points
        zi_data = DataConversion._convert_fininf(self.z, self.LB_opt, self.UB_opt)  # Data points
        zi_eval = DataConversion._convert_fininf(zi, self.LB_opt, self.UB_opt)       # Evaluation points
        
        # Calculate R matrix with proper dimensions
        eps = np.finfo(float).eps
        R = zi_eval.reshape(-1, 1) / (zi_data.reshape(1, -1) + eps)
        
        # Get characteristics
        gc = GnosticsCharacteristics(R=R, verbose=self.verbose)
        q, q1 = gc._get_q_q1(S=self.S_opt)
        
        # Store fidelities and irrelevances
        self.fj = gc._fj(q=q, q1=q1)
        self.hj = gc._hj(q=q, q1=q1)


    def _fit_qgdf(self, plot: bool = False):
        """Fit the QGDF to the data."""
        self.logger.info("Starting QGDF fitting process.")
        try:
            
            # Step 1: Data preprocessing
            self.logger.info("Preprocessing data for QGDF fitting.")
            self.data = np.sort(self.data)
            self._estimate_data_bounds()
            self._transform_data_to_standard_domain()
            self._estimate_weights()
            
            # Step 2: Bounds estimation
            self.logger.info("Estimating initial probable bounds.")
            self._estimate_initial_probable_bounds()
            self._generate_evaluation_points()
            
            # Step 3: Get distribution function values for optimization
            self.logger.info("Getting distribution function values for optimization.")
            self.df_values = self._get_distribution_function_values(use_wedf=self.wedf)
            
            # Step 4: Parameter optimization
            self.logger.info("Optimizing QGDF parameters.")
            self._determine_optimization_strategy(egdf=False)  # NOTE for QGDF egdf is False
            
            # Step 5: Calculate final QGDF and PDF
            self.logger.info("Calculating final QGDF and PDF with optimized parameters.")
            self._calculate_final_results()
            
            # Step 6: Generate smooth curves for plotting and analysis
            self.logger.info("Generating smooth curves for QGDF and PDF.")
            self._generate_smooth_curves()
            
            # Step 7: Transform bounds back to original domain
            self.logger.info("Transforming bounds back to original domain.")
            self._transform_bounds_to_original_domain()
            # Mark as fitted (Step 8 is now optional via marginal_analysis())
            self._fitted = True

            # Step 8: Z0 estimate with Z0Estimator
            self.logger.info("Estimating Z0 point with Z0Estimator.")
            self._compute_z0(optimize=self.z0_optimize) 
            # derivatives calculation
            # self._calculate_all_derivatives()
                        
            self.logger.info("QGDF fitting completed successfully.")

            if plot:
                self.logger.info("Plotting QGDF and PDF.")
                self._plot()

            # clean up computation cache
            if self.flush:  
                self.logger.info("Cleaning up computation cache.")
                self._cleanup_computation_cache()
                
        except Exception as e:
            error_msg = f"QGDF fitting failed: {e}"
            self.logger.error(error_msg)
            self.params['errors'].append({
                'method': '_fit_QGDF',
                'error': error_msg,
                'exception_type': type(e).__name__
            })
            
            self.logger.error(f"Error during QGDF fitting: {e}")
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
        self.logger.info("Computing Z0 point using Z0Estimator.")

        if self.z is None:
            self.logger.error("Data must be transformed (self.z) before Z0 estimation.")
            raise ValueError("Data must be transformed (self.z) before Z0 estimation.")
        
        # Use provided optimize parameter or fall back to instance setting
        use_optimize = optimize if optimize is not None else self.z0_optimize
        
        self.logger.info('QGDF: Computing Z0 point using Z0Estimator...')

        try:
            # Create Z0Estimator instance with proper constructor signature
            z0_estimator = Z0Estimator(
                gdf_object=self,  # Pass the QGDF object itself
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
            self.logger.info(f'QGDF: Z0 point computed successfully, (method: {method_used})')

        except Exception as e:
            # Log the error
            error_msg = f"Z0 estimation failed: {str(e)}"
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
        self.logger.info("Analyzing Z0 estimation results.")
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
        self.logger.info("Calculating all QGDF derivatives.")
        if not self._fitted:
            self.logger.error("Must fit QGDF before calculating derivatives.")
            raise RuntimeError("Must fit QGDF before calculating derivatives.")

        try:
            # Calculate derivatives using analytical methods
            second_deriv = self._get_qgdf_second_derivative()
            third_deriv = self._get_qgdf_third_derivative()
            fourth_deriv = self._get_qgdf_fourth_derivative()

            # Store in params
            if self.catch:
                self.params.update({
                    'second_derivative': second_deriv.copy(),
                    'third_derivative': third_deriv.copy(),
                    'fourth_derivative': fourth_deriv.copy()
                })
            
            self.logger.info("QGDF derivatives calculated and stored successfully.")
                
        except Exception as e:
            # Log error
            error_msg = f"Derivative calculation failed: {e}"
            self.logger.error(error_msg)
            self.params['errors'].append({
                'method': '_calculate_all_derivatives',
                'error': error_msg,
                'exception_type': type(e).__name__
            })
            self.logger.warning(f"Could not calculate derivatives: {e}")
