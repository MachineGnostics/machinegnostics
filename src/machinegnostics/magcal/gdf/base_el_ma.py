'''
ELDF Marginal Analysis Module
Marginal Cluster Analysis

Machine Gnostics
Author: Nirmal Parmar
'''

import numpy as np
import warnings
from machinegnostics.magcal.gdf.eldf import ELDF
from machinegnostics.magcal.gdf.homogeneity import DataHomogeneity

class BaseMarginalAnalysisELDF:
    '''Base class for ELDF (Estimating Local Distribution Function) Marginal Analysis.

    To perform Marginal Cluster Analysis on the ELDF.

    - estimate cluster centers, and it's boundaries
    - estimate main cluster
    - estimate lower cluster and upper cluster
    '''

    def __init__(self,
                data: np.ndarray,
                early_stopping_steps: int = 10,
                cluster_threshold: float = 0.05,
                get_clusters: bool = True,
                DLB: float = None,
                DUB: float = None,
                LB: float = None,
                UB: float = None,
                S = 'auto',
                z0_optimize: bool = True,
                tolerance: float = 1e-6,
                data_form: str = 'a',
                n_points: int = 1000,
                homogeneous: bool = True,
                catch: bool = True,
                weights: np.ndarray = None,
                wedf: bool = True,
                opt_method: str = 'L-BFGS-B',
                verbose: bool = False,
                max_data_size: int = 1000,
                flush: bool = True):
        
        self.data = data
        self.early_stopping_steps = early_stopping_steps
        self.cluster_threshold = cluster_threshold
        self.get_clusters = get_clusters
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
        self.weights = weights
        self.wedf = wedf
        self.opt_method = opt_method
        self.verbose = verbose
        self.max_data_size = max_data_size
        self.flush = flush
        # estimate cluster bounds MA ELDF default True
        self.estimate_cluster_bounds = True
        self.params = {}


        # fit status check
        self._fitted = False

        # derivative sample bound tolerance
        self._TOLERANCE = tolerance
        self._EARLY_STOPPING_STEPS = early_stopping_steps
        self._EARLY_STOPPING_THRESHOLD = 0.01  # threshold for early stopping

        # validate input parameters
        self._input_validation()

    def _input_validation(self):
        """Validate input parameters for EGDF."""

        # # max iterations
        # if not isinstance(self.max_iterations, int):
        #     raise ValueError(f"Maximum iterations must be an integer. Current type: {type(self.max_iterations)}.")

        # if self.max_iterations <= 0:
        #     raise ValueError(f"Maximum iterations must be greater than 0. Current value: {self.max_iterations}.")
        
        # early stopping steps
        if not isinstance(self.early_stopping_steps, int):
            raise ValueError(f"Early stopping steps must be an integer. Current type: {type(self.early_stopping_steps)}.")

        if self.early_stopping_steps <= 0:
            raise ValueError(f"Early stopping steps must be greater than 0. Current value: {self.early_stopping_steps}.")

        # # estimating rate
        # if not isinstance(self.estimating_rate, (int, float)):
        #     raise ValueError(f"Estimating rate must be a number. Current type: {type(self.estimating_rate)}.")

        # if self.estimating_rate <= 0:
        #     raise ValueError(f"Estimating rate must be greater than 0. Current value: {self.estimating_rate}.")
        
        # get clusters
        if not isinstance(self.get_clusters, bool):
            raise ValueError(f"Get clusters must be a boolean. Current type: {type(self.get_clusters)}.")
    
        # data
        if not isinstance(self.data, np.ndarray):
            raise ValueError(f"Data must be a numpy array. Current type: {type(self.data)}.")

        if self.data.size == 0:
            raise ValueError("Data cannot be empty.")
        
        if self.data.ndim != 1:
            raise ValueError(f"Data must be a 1-dimensional array. Current dimensions: {self.data.ndim}.")
        
    def _get_init_eldf_fit(self):
        self.init_eldf = ELDF(data=self.data,
                                  DLB=self.DLB,
                                  DUB=self.DUB,
                                  LB=self.LB,
                                  UB=self.UB,
                                  S=self.S,
                                  z0_optimize=self.z0_optimize,
                                  tolerance=self.tolerance,
                                  data_form=self.data_form,
                                  n_points=self.n_points,
                                  homogeneous=self.homogeneous,
                                  catch=self.catch,
                                  weights=self.weights,
                                  wedf=self.wedf,
                                  opt_method=self.opt_method,
                                  verbose=self.verbose,
                                  max_data_size=self.max_data_size,
                                  flush=self.flush)
        # fit init eldf model
        self.init_eldf.fit(plot=False)
        # saving bounds from initial ELDF
        self.LB = self.init_eldf.LB
        self.UB = self.init_eldf.UB
        self.DLB = self.init_eldf.DLB
        self.DUB = self.init_eldf.DUB
        self.S_opt = self.init_eldf.S_opt
        self.z0 = self.init_eldf.z0
        # store if catch is True
        if self.catch:
            self.params = self.init_eldf.params.copy()

    # # will required in interval analysis
    # def _create_extended_eldf(self, datum):
    #     data_extended = np.append(self.init_eldf.data, datum)
    #     eldf_extended = ELDF(data=data_extended,
    #                          z0_optimize=self.z0_optimize, # NOTE ELDF specific
    #                          tolerance=self.tolerance,
    #                          data_form=self.data_form,
    #                          n_points=self.n_points,
    #                          homogeneous=self.homogeneous,
    #                          catch=self.catch,
    #                          weights=self.weights,
    #                          wedf=self.wedf,
    #                          opt_method=self.opt_method,
    #                          verbose=False,  # suppress verbose during iterations
    #                          max_data_size=self.max_data_size,
    #                          flush=True)
    #     eldf_extended.fit(plot=False)    

    #     return eldf_extended

    def _get_derivatives_at_point(self, eldf_extended: ELDF, datum: float):
        """Get first, second, and third derivatives at the specified datum point."""
        # Find index of the datum in the extended data
        idx = np.where(eldf_extended.data == datum)[0][0]
        
        # Calculate derivatives
        d1 = eldf_extended.pdf[idx]  # First derivative is PDF
        d2 = eldf_extended._get_eldf_second_derivative()[idx]
        d3 = eldf_extended._get_eldf_third_derivative()[idx]
        
        return {
            'first': d1,
            'second': d2,
            'third': d3,
            'index': idx
        }

    # homogeneity check for ELDF
    def _is_homogeneous(self):
        """
        Check if the data is homogeneous.
        Returns True if homogeneous, False otherwise.
        """
        self.ih = DataHomogeneity(self.init_eldf, 
                             catch=self.catch, 
                             verbose=self.verbose)
        is_homogeneous = self.ih.test_homogeneity(estimate_cluster_bounds=self.estimate_cluster_bounds) # NOTE set true as default because we want to get cluster bounds in marginal analysis
        # cluster bounds
        self.CLB = self.ih.CLB
        self.CUB = self.ih.CUB

        if self.catch:
            self.params.update(self.ih.params)

        return is_homogeneous

    def _get_cluster(self):
        """
        Get the bounds for the clusters in the ELDF.
        """
        if self.get_clusters:
            # clusters
            lower_cluster, main_cluster, upper_cluster = self.ih.get_cluster_data()
        else:
            lower_cluster, main_cluster, upper_cluster = (None, None, None)

        return lower_cluster, main_cluster, upper_cluster

    def _plot_eldf(self, plot_type: str = 'marginal', plot_smooth: bool = True, bounds: bool = True, derivatives: bool = False, figsize: tuple = (12, 8)):
        """
        Enhanced plotting for ELDF marginal analysis with cluster bounds and other available bounds.
        
        Parameters:
        -----------
        plot_type : str, default='marginal'
            Type of plot: 'marginal', 'eldf', 'pdf', 'both'
        plot_smooth : bool, default=True
            Whether to plot smooth curves
        bounds : bool, default=True
            Whether to show bounds (LB, UB, DLB, DUB, CLB, CUB)
        derivatives : bool, default=False
            Whether to show derivative analysis plot
        figsize : tuple, default=(12, 8)
            Figure size
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # Only if self._fitted is True
        if not self._fitted:
            raise RuntimeError("Must fit marginal analysis before plotting.")
        
        if not self.catch:
            print("Plot is not available with argument catch=False")
            return
        
        if not hasattr(self.init_eldf, '_fitted') or not self.init_eldf._fitted:
            raise RuntimeError("Must fit marginal analysis before plotting.")
        
        # Validate plot_type
        valid_types = ['marginal', 'eldf', 'pdf', 'both']
        if plot_type not in valid_types:
            raise ValueError(f"plot_type must be one of {valid_types}")
        
        if derivatives:
            self._plot_derivatives_eldf(figsize=figsize)
            return
        
        # Create single plot with dual y-axes
        fig, ax1 = plt.subplots(figsize=figsize)
        
        # Get data
        x_points = self.init_eldf.data
        eldf_vals = self.init_eldf.params.get('eldf')
        pdf_vals = self.init_eldf.params.get('pdf')
        wedf_vals = self.init_eldf.params.get('wedf')
        
        # Plot ELDF on primary y-axis
        if plot_type in ['marginal', 'eldf', 'both']:
            if plot_smooth and hasattr(self.init_eldf, 'eldf_points') and self.init_eldf.eldf_points is not None:
                ax1.plot(x_points, eldf_vals, 'o', color='blue', label='ELDF', markersize=4)
                ax1.plot(self.init_eldf.di_points_n, self.init_eldf.eldf_points, 
                        color='blue', linestyle='-', linewidth=2, alpha=0.8)
            else:
                ax1.plot(x_points, eldf_vals, 'o-', color='blue', label='ELDF', 
                        markersize=4, linewidth=2, alpha=0.8)
        
        # Plot WEDF if available
        if wedf_vals is not None and plot_type in ['marginal', 'eldf', 'both']:
            ax1.plot(x_points, wedf_vals, 's', color='lightblue', 
                    label='WEDF', markersize=3, alpha=0.8)
        
        ax1.set_xlabel('Data Points')
        ax1.set_ylabel('ELDF', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_ylim(0, 1)
        
        # Create secondary y-axis for PDF
        ax2 = ax1.twinx()
        
        if plot_type in ['marginal', 'pdf', 'both']:
            if plot_smooth and hasattr(self.init_eldf, 'pdf_points') and self.init_eldf.pdf_points is not None:
                ax2.plot(x_points, pdf_vals, 'o', color='red', label='PDF', markersize=4)
                ax2.plot(self.init_eldf.di_points_n, self.init_eldf.pdf_points, 
                        color='red', linestyle='-', linewidth=2, alpha=0.8)
                max_pdf = np.max(self.init_eldf.pdf_points)
            else:
                ax2.plot(x_points, pdf_vals, 'o-', color='red', label='PDF', 
                        markersize=4, linewidth=2, alpha=0.8)
                max_pdf = np.max(pdf_vals) if pdf_vals is not None else 1
        
        ax2.set_ylabel('PDF', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        if 'max_pdf' in locals():
            ax2.set_ylim(0, max_pdf * 1.1)
        
        # Add bounds only if bounds=True
        if bounds:
            self._add_bounds_eldf(ax1)
        
        # Add marginal points (Z0 always, others only if bounds=True)
        self._add_marginal_points_eldf(ax1, bounds=bounds)
        
        # Set xlim to DLB-DUB range
        if hasattr(self.init_eldf, 'DLB') and hasattr(self.init_eldf, 'DUB'):
            # 5% data pad on either side
            pad = (self.init_eldf.DUB - self.init_eldf.DLB) * 0.05
            ax1.set_xlim(self.init_eldf.DLB - pad, self.init_eldf.DUB + pad)
            ax2.set_xlim(self.init_eldf.DLB - pad, self.init_eldf.DUB + pad)

        # Add shaded regions for bounds only if bounds=True
        if bounds and hasattr(self.init_eldf, 'LB') and hasattr(self.init_eldf, 'UB'):
            if self.init_eldf.LB is not None:
                ax1.axvspan(self.init_eldf.DLB, self.init_eldf.LB, alpha=0.15, color='purple')
            if self.init_eldf.UB is not None:
                ax1.axvspan(self.init_eldf.UB, self.init_eldf.DUB, alpha=0.15, color='brown')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Set title
        plt.title('ELDF and PDF with Bounds and Marginal Points', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def _add_marginal_points_eldf(self, ax, bounds=True):
        """Add marginal analysis points to ELDF plot."""
        marginal_info = []
        
        # Always add Z0 regardless of bounds setting
        if hasattr(self, 'params') and 'z0' in self.params:
            marginal_info.append((self.params['z0'], 'magenta', '-.', 'Z0'))
        
        # Only add other marginal points if bounds=True
        if bounds:
            # Note: No LSB/USB for ELDF marginal analysis as mentioned
            
            # Add CLB and CUB (Cluster Lower Bound and Cluster Upper Bound)
            # Keep same color scheme as EGDF (orange for cluster bounds)
            if hasattr(self, 'CLB') and self.CLB is not None:
                marginal_info.append((self.CLB, 'orange', '--', 'CLB'))
            if hasattr(self, 'CUB') and self.CUB is not None:
                marginal_info.append((self.CUB, 'orange', '--', 'CUB'))

        for point, color, style, name in marginal_info:
            # Make CLB, CUB, and Z0 lines very thin as in EGDF version
            linewidth = 1 if name in ['CLB', 'CUB', 'Z0'] else 2
            alpha = 0.6 if name in ['CLB', 'CUB', 'Z0'] else 0.8
            
            ax.axvline(x=point, color=color, linestyle=style, linewidth=linewidth, 
                    alpha=alpha, label=f"{name}={point:.3f}")

    def _add_bounds_eldf(self, ax):
        """Add bound lines to ELDF plot."""
        bound_info = []
        
        if hasattr(self.init_eldf, 'DLB') and self.init_eldf.DLB is not None:
            bound_info.append((self.init_eldf.DLB, 'green', '-', 'DLB'))
        if hasattr(self.init_eldf, 'DUB') and self.init_eldf.DUB is not None:
            bound_info.append((self.init_eldf.DUB, 'orange', '-', 'DUB'))
        if hasattr(self.init_eldf, 'LB') and self.init_eldf.LB is not None:
            bound_info.append((self.init_eldf.LB, 'purple', '--', 'LB'))
        if hasattr(self.init_eldf, 'UB') and self.init_eldf.UB is not None:
            bound_info.append((self.init_eldf.UB, 'brown', '--', 'UB'))
        
        for bound, color, style, name in bound_info:
            ax.axvline(x=bound, color=color, linestyle=style, linewidth=1, 
                    alpha=0.6, label=f"{name}={bound:.3f}")

    def _plot_derivatives_eldf(self, figsize: tuple = (14, 10)):
        """Plot ELDF derivatives for marginal analysis visualization."""
        import matplotlib.pyplot as plt
        
        if not self.catch:
            print("Plot is not available with argument catch=False")
            return
            
        if not hasattr(self.init_eldf, '_fitted') or not self.init_eldf._fitted:
            raise RuntimeError("Must fit ELDF before plotting derivatives.")
        
        # Get derivatives from init_eldf
        try:
            d1_vals = self.init_eldf.pdf
            d2_vals = self.init_eldf._get_eldf_second_derivative()
            d3_vals = self.init_eldf._get_eldf_third_derivative()
        except:
            print("Error: Could not calculate derivatives")
            return
        
        # Create subplot figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Plot derivatives
        ax1.plot(self.init_eldf.data, d1_vals, 'b-', linewidth=2, label='1st Derivative (PDF)')
        ax1.set_title('First Derivative (PDF)')
        ax1.set_ylabel('PDF Value')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.plot(self.init_eldf.data, d2_vals, 'r-', linewidth=2, label='2nd Derivative')
        ax2.set_title('Second Derivative')
        ax2.set_ylabel('2nd Derivative Value')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        ax3.plot(self.init_eldf.data, d3_vals, 'g-', linewidth=2, label='3rd Derivative')
        ax3.set_title('Third Derivative')
        ax3.set_xlabel('Data Points')
        ax3.set_ylabel('3rd Derivative Value')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Combined derivatives for boundary analysis
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.set_title('Derivative Analysis for ELDF Marginal Analysis')
        ax4.set_xlabel('Data Points')
        ax4.set_ylabel('Derivative Value')
        ax4.grid(True, alpha=0.3)
        
        # For derivatives plot, always show all marginal points
        for ax in [ax1, ax2, ax3, ax4]:
            self._add_marginal_points_eldf(ax, bounds=True)
        
        plt.suptitle('ELDF Derivative Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def _fit_eldf(self, plot:bool = False):
        # fit init eldf and get z0
        self._get_init_eldf_fit()

        # homogeneity check, pick counts, cluster bounds
        self.is_homogeneous = self._is_homogeneous()

        # get main cluster and other clusters
        self.lower_cluster, self.main_cluster, self.upper_cluster = self._get_cluster()
        

        # fit status update
        self._fitted = True
        
        # optional plot
        if plot:
            self._plot_eldf()
