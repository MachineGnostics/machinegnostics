'''
base ELDF class
Estimating Local Distribution Functions

Author: Nirmal Parmar
Machine Gnostics
'''

import numpy as np
import warnings
import logging
from machinegnostics.magcal.util.logging import get_logger
from scipy.optimize import minimize
from typing import Dict, Any
from machinegnostics.magcal.characteristics import GnosticsCharacteristics
from machinegnostics.magcal.data_conversion import DataConversion
from machinegnostics.magcal.gdf.base_egdf import BaseEGDF
from machinegnostics.magcal.scale_param import ScaleParam
from machinegnostics.magcal.gdf.z0_estimator import Z0Estimator

class BaseELDF(BaseEGDF):
    '''Base ELDF class'''
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
        self.varS = varS # ELDF specific
        self.z0_optimize = z0_optimize # ELDF specific
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

        # logger
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if verbose else logging.WARNING)
        self.logger.debug(f"{self.__class__.__name__} initialized:")

        # if S is float or int and is greater than 2, warn user
        if (isinstance(self.S, float) or isinstance(self.S, int)) and self.S > 2:
            self.logger.warning("S is greater than 2, which may not suitable for local distribution estimation. Consider using in range [0, 2]")
            warnings.warn("S is greater than 2, which may not suitable for local distribution estimation. Consider using in range [0, 2]", UserWarning)

        

    def _fit_eldf(self, plot: bool = True):
        """Fit the ELDF model to the data."""
        self.logger.info("Starting ELDF fitting process...")
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
            self.logger.info("Getting distribution function values...")
            self.df_values = self._get_distribution_function_values(use_wedf=self.wedf)
            
            # Step 4: Parameter optimization
            self.logger.info("Optimizing parameters...")
            self._determine_optimization_strategy()
            
            # Step 5: Calculate final ELDF and PDF
            self.logger.info("Computing final ELDF and PDF values...")
            self._compute_final_results()
            
            # Step 6: Generate smooth curves for plotting and analysis
            self.logger.info("Generating smooth curves for analysis...")
            self._generate_smooth_curves()
            
            # Step 7: Transform bounds back to original domain
            
            self._transform_bounds_to_original_domain()
            self.logger.info("Transformed bounds back to original data domain.")
            # Mark as fitted (Step 8 is now optional via marginal_analysis())
            self._fitted = True

            # Step 8: Z0 estimate with Z0Estimator
            self.logger.info("Estimating Z0...")
            self._compute_z0(optimize=self.z0_optimize)

            # step 9: varS
            if self.varS:
                self.logger.info("Estimating varying S...")
                self._varS_calculation()
                self.logger.info("Computing final results for varying S...")
                self._compute_final_results_varS()
                self.logger.info("Generating smooth curves for varying S...")
                self._generate_smooth_curves_varS()

            # Step 10: Z0 re-estimate with varS if enabled
            if self.varS:
                self.logger.info("Re-estimating Z0 with varying S...")
                self._compute_z0(optimize=self.z0_optimize)         
            
            self.logger.info("ELDF fitting completed successfully.")

            if plot:
                self.logger.info("Plotting results...")
                self._plot()
            
            # clean up computation cache
            if self.flush:  
                self.logger.info("Cleaning up computation cache...")
                self._cleanup_computation_cache()
                    
        except Exception as e:
            # log error
            error_msg = f"ELDF fitting failed: {e}"
            self.logger.error(error_msg)
            self.params['errors'].append({
                'method': '_fit_eldf',
                'error': error_msg,
                'exception_type': type(e).__name__
            })
            self.logger.info(f"Error during ELDF fitting: {e}")
            raise e


    def _compute_eldf_core(self, S, LB, UB, zi_data=None, zi_eval=None):
        """Core computation for the ELDF model."""
        self.logger.info("Computing core ELDF values...")
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
        return self._estimate_eldf_from_moments(fi, hi), fi, hi

    def _estimate_eldf_from_moments(self, fidelity, irrelevance):
        """Estimate the ELDF from moments."""
        self.logger.info("Estimating ELDF from moments...")
        weights = self.weights.reshape(-1, 1)

        mean_irrelevance = np.sum(weights * irrelevance, axis=0) / np.sum(weights)

        eldf_values = (1 - mean_irrelevance) / 2

        return eldf_values.flatten()

    def _compute_final_results(self):
        """Compute the final results for the ELDF model."""
        self.logger.info("Computing final ELDF and PDF results...")
        # Implement final results computation logic here
        zi_d = DataConversion._convert_fininf(self.z, self.LB_opt, self.UB_opt)
        self.zi = zi_d

        # Calculate ELDF and get moments
        eldf_values, fi, hi = self._compute_eldf_core(self.S_opt, self.LB_opt, self.UB_opt)

        # Store for derivative calculations
        self.fi = fi
        self.hi = hi

        self.eldf = eldf_values
        self.pdf = self._compute_eldf_pdf(self.fi, self.hi)
        
        if self.catch:
            self.logger.info("Catching parameters...")
            self.params.update({
                'eldf': self.eldf.copy(),
                'pdf': self.pdf.copy(),
                'zi': self.zi.copy(),
                # 'S_var': self.S_var.copy() if self.varS else None
            })

    def _compute_eldf_pdf(self, fi, hi):
        """Compute the PDF for the ELDF model."""
        self.logger.info("Computing PDF from ELDF moments...")
        weights = self.weights.reshape(-1, 1)

        # fi_mean
        fi_mean = np.sum(weights * fi, axis=0) / np.sum(weights)
        pdf_values = ((fi_mean)**2)/(self.S_opt)
        return pdf_values.flatten()

    def _generate_smooth_curves(self):
        """Generate smooth curves for plotting and analysis - ELDF."""
        self.logger.info("Generating smooth curves for ELDF...")
        try:
            self.logger.info("Generating smooth curves without varying S...")

            smooth_eldf, self.smooth_fi, self.smooth_hi = self._compute_eldf_core(
                self.S_opt, self.LB_opt, self.UB_opt,
                zi_data=self.z_points_n, zi_eval=self.z
            )
            smooth_pdf = self._compute_eldf_pdf(self.smooth_fi, self.smooth_hi) 
        
            self.eldf_points = smooth_eldf
            self.pdf_points = smooth_pdf
            
            # Store zi_n for derivative calculations
            self.zi_n = DataConversion._convert_fininf(self.z_points_n, self.LB_opt, self.UB_opt)
            
            # Mark as generated
            self._computation_cache['smooth_curves_generated'] = True
            
            if self.catch:
                self.logger.info("Catching parameters...")
                self.params.update({
                    'eldf_points': self.eldf_points.copy(),
                    'pdf_points': self.pdf_points.copy(),
                    'zi_points': self.zi_n.copy()
                })

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

            self.logger.warning(f"Could not generate smooth curves: {e}")

            # Create fallback points using original data
            self.eldf_points = self.eldf.copy() if hasattr(self, 'eldf') else None
            self.pdf_points = self.pdf.copy() if hasattr(self, 'pdf') else None
            self._computation_cache['smooth_curves_generated'] = False

    
    def _plot(self, plot_smooth: bool = True, plot: str = 'both', bounds: bool = True, extra_df: bool = True, figsize: tuple = (12, 8)):
        """Enhanced plotting with better organization."""
        self.logger.info("Preparing to plot ELDF and PDF...")
        import matplotlib.pyplot as plt
    
        if plot_smooth and (len(self.data) > self.max_data_size) and self.verbose:
            self.logger.warning(f"Given data size ({len(self.data)}) exceeds max_data_size ({self.max_data_size}). For optimal compute performance, set 'plot_smooth=False', or 'max_data_size' to a larger value whichever is appropriate.")

        if not self.catch:
            self.logger.warning("Plot is not available with argument catch=False")
            return
        
        if not self._fitted:
            raise RuntimeError("Must fit ELDF before plotting.")
        
        # Validate plot parameter
        if plot not in ['gdf', 'pdf', 'both']:
            raise ValueError("plot parameter must be 'gdf', 'pdf', or 'both'")
        
        # Check data availability
        if plot in ['gdf', 'both'] and self.params.get('eldf') is None:
            raise ValueError("ELDF must be calculated before plotting GDF")
        if plot in ['pdf', 'both'] and self.params.get('pdf') is None:
            raise ValueError("PDF must be calculated before plotting PDF")
        
        # Prepare data
        x_points = self.data
        eldf_plot = self.params.get('eldf')
        pdf_plot = self.params.get('pdf')
        wedf = self.params.get('wedf')
        ksdf = self.params.get('ksdf')
        
        # Check smooth plotting availability
        has_smooth = (hasattr(self, 'z_points_n') and hasattr(self, 'eldf_points') 
                     and hasattr(self, 'pdf_points') and self.z_points_n is not None
                     and self.eldf_points is not None and self.pdf_points is not None)
        plot_smooth = plot_smooth and has_smooth
        
        # Create figure
        fig, ax1 = plt.subplots(figsize=figsize)
    
        # Plot ELDF if requested
        if plot in ['gdf', 'both']:
            self._plot_eldf(ax1, x_points, eldf_plot, plot_smooth, extra_df, wedf, ksdf)
        
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
        self.logger.info("Plotting PDF...")
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

    def _plot_eldf(self, ax, x_points, eldf_plot, plot_smooth, extra_df, wedf, ksdf):
        """Plot ELDF components."""
        self.logger.info("Plotting ELDF...")
        if plot_smooth and hasattr(self, 'eldf_points') and self.eldf_points is not None:
            ax.plot(x_points, eldf_plot, 'o', color='blue', label='ELDF', markersize=4)
            ax.plot(self.di_points_n, self.eldf_points, color='blue', 
                linestyle='-', linewidth=2, alpha=0.8)
        else:
            ax.plot(x_points, eldf_plot, 'o-', color='blue', label='ELDF', 
                markersize=4, linewidth=1, alpha=0.8)
        
        if extra_df:
            if wedf is not None:
                ax.plot(x_points, wedf, 's', color='lightblue', 
                    label='WEDF', markersize=3, alpha=0.8)
            if ksdf is not None:
                ax.plot(x_points, ksdf, 's', color='cyan', 
                    label='KS Points', markersize=3, alpha=0.8)
        
        ax.set_ylabel('ELDF', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax.set_ylim(0, 1)

    def _add_plot_formatting(self, ax1, plot, bounds):
        """Add formatting, bounds, and legends to plot."""
        self.logger.info("Adding plot formatting and bounds...")
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
            'gdf': 'ELDF' + (' with Bounds' if bounds else ''),
            'pdf': 'PDF' + (' with Bounds' if bounds else ''),
            'both': 'ELDF and PDF' + (' with Bounds' if bounds else '')
        }
        
        ax1.set_title(titles[plot])
        ax1.legend(loc='upper left', bbox_to_anchor=(0, 1))
        ax1.grid(True, alpha=0.3)

    def _get_eldf_second_derivative(self, fi, hi):
        """Calculate second derivative of ELDF from stored fidelities and irrelevances."""
        self.logger.info("Calculating second derivative of ELDF...")
        if fi is None or hi is None:
            fi = self.fi
            hi = self.hi

        if fi is None or hi is None:
            raise ValueError("Fidelities and irrelevances must be calculated before second derivative estimation.")
        
        weights = self.weights.reshape(-1, 1)
        
        # Calculate f^2 * h (fidelity squared times irrelevance)
        fh = (self.fi**2) * self.hi
        
        # Weight and scale by S^2
        weighted_fh = fh * (weights / (self.S_opt**2))
        
        # Sum over data points
        second_derivative = 4 * np.sum(weighted_fh, axis=0) / np.sum(weights)
        
        return second_derivative.flatten()
    
    def _get_eldf_third_derivative(self, fi, hi):
        """Calculate third derivative of ELDF from stored fidelities and irrelevances."""
        self.logger.info("Calculating third derivative of ELDF...")
        if fi is None or hi is None:
                fi = self.fi
                hi = self.hi

        if fi is None or hi is None:
            raise ValueError("Fidelities and irrelevances must be calculated before third derivative estimation.")
        
        weights = self.weights.reshape(-1, 1)
        
        # Calculate components
        h2 = self.hi**2  # h^2
        f2 = self.fi**2  # f^2
        f2h2 = f2 * h2   # f^2 * h^2
        f4 = f2**2       # f^4
        
        # Calculate the expression: 8 * (2 * f^2 * h^2 - f^4) * (W / S^3)
        expression = 2 * f2h2 - f4
        weighted_expression = expression * (weights / (self.S_opt**3))
        
        # Sum over data points
        third_derivative = 8 * np.sum(weighted_expression, axis=0) / np.sum(weights)
        
        return third_derivative.flatten()
    
    def _get_eldf_fourth_derivative(self, fi, hi):
        """Calculate fourth derivative of ELDF from stored fidelities and irrelevances."""
        self.logger.info("Calculating fourth derivative of ELDF...")
        if fi is None or hi is None:
            fi = self.fi
            hi = self.hi

        if self.fi is None or self.hi is None:
            raise ValueError("Fidelities and irrelevances must be calculated before fourth derivative estimation.")
        
        weights = self.weights.reshape(-1, 1)
        
        # Calculate components
        f2 = self.fi**2  # f^2
        h = self.hi      # h
        h3 = h**3        # h^3
        
        # Calculate f^2 * h^3 and f^4 * h
        f2h3 = f2 * h3
        f4h = (f2**2) * h  # f^4 * h
        
        # Weight and scale by (2/S)^4
        scale_factor = (2 / self.S_opt)**4
        weighted_f2h3 = f2h3 * (weights * scale_factor)
        weighted_f4h = f4h * (weights * scale_factor)
        
        # Calculate the expression: 4 * (f^2 * h^3 - 2 * f^4 * h)
        sum_f2h3 = np.sum(weighted_f2h3, axis=0) / np.sum(weights)
        sum_f4h = np.sum(weighted_f4h, axis=0) / np.sum(weights)
        
        fourth_derivative = 4 * (sum_f2h3 - 2 * sum_f4h)
        
        return fourth_derivative.flatten()
    
    def _get_results(self)-> dict:
        """Return fitting results."""
        self.logger.info("Retrieving ELDF fitting results...")
        if not self._fitted:
            raise RuntimeError("Must fit ELDF before getting results.")
        
        # selected key from params if exists
        keys = ['DLB', 'DUB', 'LB', 'UB', 'S_opt', 'z0', 'eldf', 'pdf',
                'eldf_points', 'pdf_points', 'zi', 'zi_points', 'weights']
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
        self.logger.info("Computing Z0 point using Z0Estimator...")
        if self.z is None:
            self.logger.error("Data must be transformed (self.z) before Z0 estimation.")
            raise ValueError("Data must be transformed (self.z) before Z0 estimation.")
        
        # Use provided optimize parameter or fall back to instance setting
        use_optimize = optimize if optimize is not None else self.z0_optimize

        self.logger.info('ELDF: Computing Z0 point using Z0Estimator...')

        try:
                # Create Z0Estimator instance with proper constructor signature
            z0_estimator = Z0Estimator(
                gdf_object=self,  # Pass the ELDF object itself
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
            self.logger.info(f'ELDF: Z0 point computed successfully, (method: {method_used})')

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
        self.logger.info("Using fallback method for Z0 point...")

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
        self.logger.info("Analyzing Z0 estimation results...")
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
    
    def _calculate_fidelities_irrelevances_at_given_zi(self, zi):
        """Helper method to recalculate fidelities and irrelevances for current zi."""
        self.logger.info("Calculating fidelities and irrelevances at given zi...")
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
        self.logger.info("Calculating varying S (varS)...")
        from machinegnostics import variance

        self.logger.info("Calculating varS for ELDF...")
        # estimate fi hi at z0
        gc, q, q1 = self._calculate_gcq_at_given_zi(self.z0)

        fi_z0 = gc._fi(q=q, q1=q1)

        scale = ScaleParam()
        self.S_local = scale._gscale_loc(fi_z0)

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
        """Compute the final results for the ELDF model."""
        self.logger.info("Computing final ELDF and PDF results with varying S...")
        # Implement final results computation logic here
        # zi_d = DataConversion._convert_fininf(self.z, self.LB_opt, self.UB_opt)
        # self.zi = zi_d

        eldf_values, fi, hi = self._compute_eldf_core(self.S_var, self.LB_opt, self.UB_opt)
        self.fi = fi
        self.hi = hi

        self.eldf = eldf_values
        self.pdf = self._compute_eldf_pdf(self.fi, self.hi)
        
        if self.catch:
            self.logger.info("Catching parameters with varying S...")
            self.params.update({
                'eldf': self.eldf.copy(),
                'pdf': self.pdf.copy(),
                'zi': self.zi.copy(),
                'S_var': self.S_var.copy() if self.varS else None
            })

    def _generate_smooth_curves_varS(self):
        """Generate smooth curves for plotting and analysis - ELDF."""
        self.logger.info("Generating smooth curves for ELDF with varying S...")
        try:
            self.logger.info("Generating smooth curves with varying S...")

            smooth_eldf, self.smooth_fi, self.smooth_hi = self._compute_eldf_core(
                self.S_var, self.LB_opt, self.UB_opt,
                zi_data=self.z_points_n, zi_eval=self.z
            )
            smooth_pdf = self._compute_eldf_pdf(self.smooth_fi, self.smooth_hi) 
        
            self.eldf_points = smooth_eldf
            self.pdf_points = smooth_pdf
            
            # Mark as generated
            self._computation_cache['smooth_curves_generated'] = True
            
            if self.catch:
                self.params.update({
                    'eldf_points': self.eldf_points.copy(),
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

            self.logger.warning(f"Could not generate smooth curves: {e}")

            # Create fallback points using original data
            self.eldf_points = self.eldf.copy() if hasattr(self, 'eldf') else None
            self.pdf_points = self.pdf.copy() if hasattr(self, 'pdf') else None
            self._computation_cache['smooth_curves_generated'] = False