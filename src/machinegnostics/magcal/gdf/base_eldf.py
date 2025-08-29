'''
base ELDF class
Estimating Local Distribution Functions

Author: Nirmal Parmar
Machine Gnostics
'''

import numpy as np
import warnings
from scipy.optimize import minimize
from machinegnostics.magcal.characteristics import GnosticsCharacteristics
from machinegnostics.magcal.gdf.base_df import BaseDistFunc
from machinegnostics.magcal.data_conversion import DataConversion
from machinegnostics.magcal.gdf.wedf import WEDF
from machinegnostics.magcal.mg_weights import GnosticsWeights
from machinegnostics.magcal.gdf.base_egdf import BaseEGDF
from machinegnostics.magcal.gdf.base_distfunc import BaseDistFuncCompute

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

        # Validate all inputs
        self._validate_inputs()
        # validation for varS, it should be a boolean
        if not isinstance(self.varS, bool):
            raise ValueError("varS must be a boolean")

        # Store initial parameters if catching
        if self.catch:
            self._store_initial_params()

    def _fit_eldf(self, plot: bool = True):
        """Fit the ELDF model to the data."""
    #     try:
        if self.verbose:
            print("Starting ELDF fitting process...")

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
        self._determine_optimization_strategy()
        
        # Step 5: Calculate final ELDF and PDF
        self._compute_final_results()
        
        # Step 6: Generate smooth curves for plotting and analysis
        self._generate_smooth_curves()
        
        # Step 7: Transform bounds back to original domain
        self._transform_bounds_to_original_domain()
        
        # Mark as fitted (Step 8 is now optional via marginal_analysis())
        self._fitted = True
        
        if self.verbose:
            print("ELDF fitting completed successfully.")

        # if plot:
        #     self._plot()
        
        # clean up computation cache
        if self.flush:  
            self._cleanup_computation_cache()
                
    #     except Exception as e:
    #         if self.verbose:
    #             print(f"Error during EGDF fitting: {e}")
    #         raise e


    def _compute_eldf_core(self, S, LB, UB, zi_data=None, zi_eval=None):
        """Core computation for the ELDF model."""
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
        fi = gc._fi(q=q, q1=q1)
        hi = gc._hi(q=q, q1=q1)
        return self._estimate_eldf_from_moments(fi, hi), fi, hi

    def _estimate_eldf_from_moments(self, fidelity, irrelevance):
        """Estimate the ELDF from moments."""
        weights = self.weights.reshape(-1, 1)

        mean_irrelevance = np.sum(weights * irrelevance, axis=0) / np.sum(weights)

        eldf_values = (1 - mean_irrelevance) / 2

        return eldf_values.flatten()

    def _compute_final_results(self):
        """Compute the final results for the ELDF model."""
        # Implement final results computation logic here
        zi_d = DataConversion._convert_fininf(self.z, self.LB_opt, self.UB_opt)
        self.zi = zi_d

        # Calculate ELDF and get moments
        eldf_values, fi, hi = self._compute_eldf_core(self.S_opt, self.LB_opt, self.UB_opt)

        # Store for derivative calculations
        self.fi = fi
        self.hi = hi
        self.eldf = eldf_values
        self.pdf = self._compute_eldf_pdf(fi, hi)
        
        if self.catch:
            self.params.update({
                'eldf': self.eldf.copy(),
                'pdf': self.pdf.copy(),
                'zi': self.zi.copy()
            })

    def _compute_eldf_pdf(self, fi, hi):
        """Compute the PDF for the ELDF model."""
        weights = self.weights.reshape(-1, 1)

        # fi_mean
        fi_mean = np.sum(weights * fi, axis=0) / np.sum(weights)
        pdf_values = ((fi_mean)**2)/(self.S_opt)
        return pdf_values.flatten()

    def _generate_smooth_curves(self):
        """Generate smooth curves for plotting and analysis - ELDF."""
        # try:
        # Generate smooth ELDF and PDF
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
            self.params.update({
                'eldf_points': self.eldf_points.copy(),
                'pdf_points': self.pdf_points.copy(),
                'zi_points': self.zi_n.copy()
            })
        
        if self.verbose:
            print(f"Generated smooth curves with {self.n_points} points.")
                
        # except Exception as e:
        #     if self.verbose:
        #         print(f"Warning: Could not generate smooth curves: {e}")

        #     # Create fallback points using original data
        #     self.eldf_points = self.eldf.copy() if hasattr(self, 'eldf') else None
        #     self.pdf_points = self.pdf.copy() if hasattr(self, 'pdf') else None
        #     self._computation_cache['smooth_curves_generated'] = False

    def _compute_varS(self):
        """Compute the varying S for the ELDF model."""
        # Implement variance computation logic here
        pass

    def fit(self):
        """Fit the ELDF model to the data."""
        self._fit_eldf()

    
    def _plot(self, plot_smooth: bool = True, plot: str = 'both', bounds: bool = True, extra_df: bool = True, figsize: tuple = (12, 8)):
        """Enhanced plotting with better organization."""
        import matplotlib.pyplot as plt
    
        if plot_smooth and (len(self.data) > self.max_data_size) and self.verbose:
            print(f"Warning: Given data size ({len(self.data)}) exceeds max_data_size ({self.max_data_size}). For optimal compute performance, set 'plot_smooth=False', or 'max_data_size' to a larger value whichever is appropriate.")
    
        if not self.catch:
            print("Plot is not available with argument catch=False")
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
        
        plt.tight_layout()
        plt.show()
    
    def _plot_pdf(self, ax, x_points, pdf_plot, plot_smooth, is_secondary=False):
        """Plot PDF components."""
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

    def plot(self, plot_smooth: bool = True, plot: str = 'both', bounds: bool = True, extra_df: bool = True, figsize: tuple = (12, 8)):
        """Public interface for plotting ELDF results."""
        self._plot(plot_smooth=plot_smooth, plot=plot, bounds=bounds, extra_df=extra_df, figsize=figsize)
