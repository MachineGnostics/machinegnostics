'''
ELDF Interval Analysis

Local Estimating Interval Analysis

Author: Nirmal Parmar
Machine Gnostics
'''

import numpy as np
import warnings
from machinegnostics.magcal import ELDF
from machinegnostics.magcal.gdf.base_el_ma import BaseMarginalAnalysisELDF
from machinegnostics.magcal.gdf.intv_engine import IntveEngine

class BaseIntervalAnalysisELDF(BaseMarginalAnalysisELDF):
    """
    Base class for ELDF Interval Analysis.
    
    This is an internal developer class that provides the foundation for performing
    interval analysis on Empirical Likelihood Distribution Functions (ELDF). It extends
    the marginal analysis capabilities with interval estimation functionality.
    
    The class handles:
    - Data homogeneity assessment and clustering
    - Initial ELDF fitting with outlier handling
    - Interval engine integration for extrema estimation
    - Comprehensive validation and user guidance
    
    Developer Notes:
    ----------------
    - This is NOT a public API class - use derived classes for public interfaces
    - Inherits from BaseMarginalAnalysisELDF for clustering and marginal analysis
    - Uses IntveEngine for the core interval estimation algorithms
    - Implements extensive validation logic with contextual warnings
    - All methods prefixed with '_' are internal and may change without notice
    
    Key Internal Workflow:
    ----------------------
    1. Initial ELDF fitting (_get_init_eldf_fit)
    2. Homogeneity assessment (_is_homogeneous)
    3. Cluster identification and validation
    4. Main cluster ELDF fitting (if needed)
    5. Interval engine execution
    6. Results extraction and storage
    
    Attributes:
    -----------
    Z0, Z0L, Z0U : float
        Interval estimation results from IntveEngine
    ZL, ZU : float
        Lower and upper interval bounds
    intv : IntveEngine
        The interval analysis engine instance
    _fitted : bool
        Internal fitting status flag
        
    Warning Categories:
    -------------------
    - Data-parameter mismatch warnings (homogeneous setting vs actual data)
    - Insufficient data warnings (cluster size, main cluster quality)
    - Configuration mismatch warnings (parameter combinations)
    
    Thread Safety:
    --------------
    Not thread-safe. Each instance should be used by a single thread.
    
    Performance Notes:
    ------------------
    - Large datasets are automatically subsampled (max_data_size parameter)
    - Clustering operations scale O(n log n) with data size
    - Interval estimation complexity depends on n_points_per_direction
    """
    
    def __init__(self,
            data: np.ndarray,
            DLB: float = None,
            DUB: float = None,
            LB: float = None,
            UB: float = None,
            S = 'auto',
            varS: bool = False,
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
            flush: bool = True,
            early_stopping_steps: int = 10,
            cluster_threshold: float = 0.05,
            estimate_cluster_bounds: bool = True,
            get_clusters: bool = True,
            n_points_per_direction: int = 1000, # intv engine specific
            dense_zone_fraction: float = 0.4,
            dense_points_fraction: float = 0.7,
            convergence_window: int = 15,
            convergence_threshold: float = 1e-7,
            min_search_points: int = 30,
            boundary_margin_factor: float = 0.001,
            extrema_search_tolerance: float = 1e-6,):
        super().__init__(data=data,
                         early_stopping_steps=early_stopping_steps,
                         cluster_threshold=cluster_threshold,
                         get_clusters=get_clusters,
                         DLB=DLB,
                         DUB=DUB,
                         LB=LB,
                         UB=UB,
                         S=S,
                         varS=varS,
                         z0_optimize=z0_optimize,
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

        # Initialize interval-specific attributes
        self.Z0 = None
        self.Z0L = None
        self.Z0U = None
        self.ZL = None
        self.ZU = None

        # Common attributes
        self.data = data
        
        self.DLB = DLB
        self.DUB = DUB
        self.LB = LB
        self.UB = UB
        self.S = S
        self.varS = varS
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
        self.estimate_cluster_bounds = estimate_cluster_bounds # ELDF intv specific
        self.early_stopping_steps = early_stopping_steps
        self.cluster_threshold = cluster_threshold
        self.get_clusters = get_clusters
        self.n_points_per_direction = n_points_per_direction
        self.dense_zone_fraction = dense_zone_fraction
        self.dense_points_fraction = dense_points_fraction
        self.convergence_window = convergence_window
        self.convergence_threshold = convergence_threshold
        self.min_search_points = min_search_points
        self.boundary_margin_factor = boundary_margin_factor
        self.extrema_search_tolerance = extrema_search_tolerance
        self.params = {}

        # fit status
        self._fitted = False

    def _fit_eldf_intv(self, plot: bool = False):
        """
        Internal method to perform complete ELDF interval analysis.
        
        Developer Notes:
        ----------------
        This is the main orchestration method that coordinates:
        1. Initial ELDF fitting and parameter extraction
        2. Data homogeneity assessment and validation
        3. Cluster analysis and main cluster identification
        4. Conditional re-fitting on main cluster
        5. Interval engine execution and result extraction
        
        Parameters:
        -----------
        plot : bool
            Whether to generate diagnostic plots during fitting
            
        Side Effects:
        -------------
        - Sets self._fitted = True upon successful completion
        - Populates interval results (Z0, Z0L, Z0U, ZL, ZU)
        - May issue warnings for data/parameter mismatches
        - Creates self.intv (IntveEngine instance)
        
        Raises:
        -------
        Various exceptions from underlying ELDF fitting or interval engine
        """
        try:
            # fit ELDF
            # extract z0, and bounds
            self._get_init_eldf_fit(plot=plot)

            # homogeneity check, pick counts, cluster bounds
            self.is_homogeneous = self._is_homogeneous()
            # user understanding check with data homogeneity
            self._homogeneity_validation_and_msg()

            # get main cluster and other clusters
            self.lower_cluster, self.main_cluster, self.upper_cluster = self._get_cluster()

            # main cluster check
            self._main_cluster_validation_and_msg()

            # if data is not homogeneous, user option
            # get the main cluster and the cluster bounds
            if not self.homogeneous and self.get_clusters and len(self.main_cluster) > 4: # + user choice
                self._get_main_init_eldf(self.main_cluster, plot=plot)

            if self.verbose:
                    print("Initiating ELDF Interval Analysis...")
                    
            # intv analysis
            # get extended ELDF with a new datum
            self.intv = IntveEngine(self.init_eldf,
                                n_points_per_direction=self.n_points_per_direction,
                                dense_zone_fraction=self.dense_zone_fraction,
                                dense_points_fraction=self.dense_points_fraction,
                                convergence_window=self.convergence_window,
                                convergence_threshold=self.convergence_threshold,
                                min_search_points=self.min_search_points,
                                boundary_margin_factor=self.boundary_margin_factor,
                                extrema_search_tolerance=self.extrema_search_tolerance,
                                verbose=self.verbose)

            self.intv.fit(plot=plot, update_df_params=True)
            # extract results
            self.Z0 = self.intv.z0
            self.Z0L = self.intv.z0l
            self.Z0U = self.intv.z0u
            self.ZL = self.intv.zl
            self.ZU = self.intv.zu

            # status update
            self._fitted = True

            # plot intv
            if plot:
                self._plot_eldf_intv(figsize=(12, 8))

            # verbose message
            if self.verbose:
                print("ELDF Interval Analysis completed.")
        except Exception as e:
            raise RuntimeError(f"ELDF Interval Analysis failed: {str(e)}")

    def _main_cluster_validation_and_msg(self):
        """
        Validates main cluster quality and provides user guidance.
        
        Developer Notes:
        ----------------
        Checks if the identified main cluster is suitable for reliable ELDF fitting.
        Issues warnings when cluster quality is insufficient, which could lead to
        unreliable interval estimates.
        
        Validation Criteria:
        - Main cluster exists (not None)  
        - Main cluster has at least 4 data points (minimum for ELDF fitting)
        
        Warning Conditions:
        - Cluster too small: < 4 points (insufficient for parameter estimation)
        - Cluster undefined: clustering algorithm failed to identify main group
        """
        # main cluster check
        if self.main_cluster is None or len(self.main_cluster) < 4:
            warnings.warn(
                "Insufficient main cluster data detected. "
                f"Main cluster has {len(self.main_cluster) if self.main_cluster is not None else 0} points, "
                "but at least 4 are required for reliable ELDF parameter estimation. "
                "This may result in unreliable interval estimates. "
                "Consider: (1) increasing data size, (2) adjusting cluster_threshold parameter, "
                "(3) setting homogeneous=True if outliers are not expected, or "
                "(4) reviewing data quality for potential issues.",
                UserWarning,
                stacklevel=2
            )
        else:
            if self.verbose:
                print(f"✓ Main cluster validated with {len(self.main_cluster)} data points.")

    def _homogeneity_validation_and_msg(self):
        """
        Validates consistency between user settings and actual data characteristics.
        
        Developer Notes:
        ----------------
        Performs cross-validation between user-specified homogeneity assumptions
        and algorithmically-determined data characteristics. Issues actionable
        warnings when mismatches are detected.
        
        Validation Logic:
        - homogeneous=True + heterogeneous data → suggest clustering approach
        - homogeneous=False + homogeneous data → suggest simplified approach  
        - Missing cluster bounds + heterogeneous data → suggest enabling estimation
        
        Warning Categories:
        - Parameter-data mismatch (most critical)
        - Suboptimal configuration (performance/accuracy)
        - Missing required settings (functionality)
        """
        # user understanding check with data homogeneity
        if self.homogeneous == True and self.is_homogeneous == False:
            warnings.warn(
                "Data-parameter mismatch detected: Your data appears heterogeneous (contains outliers/clusters), "
                "but 'homogeneous=True' was specified. This may lead to biased interval estimates. "
                "Recommended actions: (1) Set homogeneous=False and get_clusters=True for robust outlier handling, "
                "or (2) if your dataset is small (<50 points) with no clear clustering patterns, "
                "you may continue with current settings but interpret results cautiously.",
                UserWarning,
                stacklevel=2
            )
        elif self.homogeneous == False and self.is_homogeneous == True:
            warnings.warn(
                "Suboptimal configuration detected: Your data appears homogeneous (no significant outliers), "
                "but 'homogeneous=False' was specified. This adds unnecessary computational overhead. "
                "Consider setting homogeneous=True to improve performance and skip cluster analysis.",
                UserWarning,
                stacklevel=2
            )
        else:
            if self.verbose:
                print("✓ Data homogeneity setting matches detected data characteristics.")

        # Informational messages about detected data structure
        if self.is_homogeneous:
            if self.verbose:
                print("→ Data assessment: Homogeneous structure detected. Using full dataset for interval analysis.")
        else:
            if self.verbose:
                print("→ Data assessment: Heterogeneous structure detected. Cluster-based analysis required.")

        # Configuration consistency check for heterogeneous data
        if self.is_homogeneous == False and self.estimate_cluster_bounds == False and self.get_clusters == True:
            warnings.warn(
                "Incomplete configuration for heterogeneous data: 'get_clusters=True' was specified, "
                "but 'estimate_cluster_bounds=False'. For heterogeneous data, cluster boundary estimation "
                "is typically required to identify the main cluster accurately. "
                "Consider setting estimate_cluster_bounds=True for optimal results.",
                UserWarning,
                stacklevel=2
            )

    # def _plot_eldf_intv(self, figsize=(12, 8)):
    #     """
    #     Generate diagnostic plots for interval analysis results.
        
    #     Developer Notes:
    #     ----------------
    #     Creates visualization of both the interval analysis results and the
    #     underlying initial ELDF fit. Useful for debugging and result interpretation.
        
    #     Raises:
    #     -------
    #     RuntimeError : If called before fitting is complete
    #     """
    #     if not self._fitted:
    #         raise RuntimeError(
    #             "Cannot generate plots: Interval analysis not yet fitted. "
    #             "Please call the 'fit' method before attempting to plot results."
    #         )
    #     self.intv.plot(figsize=figsize)
    #     self.init_eldf.plot(figsize=figsize)

    def _get_main_init_eldf(self, cluster: np.ndarray, plot: bool = False):
        """
        Fit ELDF model specifically to the main cluster data.
        
        This method creates a new ELDF instance using only the main cluster data,
        which is essential for obtaining accurate interval estimates when the full
        dataset contains outliers or multiple clusters.
        
        Developer Notes:
        ----------------
        - Called only when data is determined to be heterogeneous
        - Updates instance bounds (LB, UB, DLB, DUB) from the cluster-specific fit
        - Preserves all user-specified fitting parameters
        - Updates internal parameter storage if catch=True
        
        Parameters:
        -----------
        cluster : np.ndarray
            Main cluster data points identified by clustering algorithm
        plot : bool
            Whether to generate diagnostic plots during ELDF fitting
            
        Returns:
        --------
        ELDF
            The fitted ELDF instance for the main cluster
            
        Side Effects:
        -------------
        - Updates self.init_eldf with cluster-specific model
        - Updates boundary parameters (LB, UB, DLB, DUB, S_opt, z0)
        - Updates self.params if catch=True
        """
        if self.verbose:
            print(f"→ Fitting specialized ELDF model to main cluster ({len(cluster)} points)...")

        self.init_eldf = ELDF(data=cluster,
                                  varS=self.varS,
                                  z0_optimize=self.z0_optimize,
                                  tolerance=self.tolerance,
                                  data_form=self.data_form,
                                  n_points=self.n_points,
                                  catch=self.catch,
                                  wedf=self.wedf,
                                  opt_method=self.opt_method,
                                  verbose=self.verbose,
                                  max_data_size=self.max_data_size,
                                  flush=self.flush)
        # fit init eldf model
        self.init_eldf.fit(plot=plot)
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
        if self.verbose:
            print("✓ Main cluster ELDF fitting completed successfully.")

        return self.init_eldf
    
    def _plot_eldf_intv(self, figsize=(12, 8)):
        """
        Generate comprehensive ELDF Interval Analysis plots with tolerance and typical data intervals.
        
        Creates visualization showing:
        - ELDF curve with distribution fitting
        - PDF curve on secondary axis
        - Tolerance interval (Z0L, Z0U) as light green filled zone
        - Typical data interval (ZL, ZU) as light blue filled zone
        - All critical points and bounds including Z0 vertical line
        - Original data points as rug plot
        
        Developer Notes:
        ----------------
        Creates visualization of both the interval analysis results and the
        underlying initial ELDF fit. Useful for debugging and result interpretation.
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 8)
            Figure size as (width, height) in inches
            
        Raises:
        -------
        RuntimeError : If called before fitting is complete
        """
        if not self._fitted:
            raise RuntimeError(
                "Cannot generate plots: Interval analysis not yet fitted. "
                "Please call the 'fit' method before attempting to plot results."
            )
        
        if not self.catch:
            print("Plot is not available with argument catch=False")
            return
            
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            raise ImportError("matplotlib and numpy required for plotting")
        
        # Create figure with dual y-axes
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = ax1.twinx()
        
        # Get data from fitted ELDF
        x_points = self.init_eldf.data
        eldf_vals = self.init_eldf.params.get('eldf') if hasattr(self.init_eldf, 'params') else None
        pdf_vals = self.init_eldf.params.get('pdf') if hasattr(self.init_eldf, 'params') else None
        wedf_vals = self.init_eldf.params.get('wedf') if hasattr(self.init_eldf, 'params') else None
        
        # Get smooth curve data if available
        smooth_x = None
        smooth_eldf = None
        smooth_pdf = None
        
        if hasattr(self.init_eldf, 'di_points_n') and self.init_eldf.di_points_n is not None:
            smooth_x = self.init_eldf.di_points_n
            
            if hasattr(self.init_eldf, 'eldf_points') and self.init_eldf.eldf_points is not None:
                smooth_eldf = self.init_eldf.eldf_points
                
            if hasattr(self.init_eldf, 'pdf_points') and self.init_eldf.pdf_points is not None:
                smooth_pdf = self.init_eldf.pdf_points
        
        # Set up x-axis range with padding
        if hasattr(self.init_eldf, 'DLB') and hasattr(self.init_eldf, 'DUB'):
            x_range = self.init_eldf.DUB - self.init_eldf.DLB
            x_pad = x_range * 0.05
            x_min = self.init_eldf.DLB - x_pad
            x_max = self.init_eldf.DUB + x_pad
        else:
            data_range = np.max(x_points) - np.min(x_points)
            x_pad = data_range * 0.05
            x_min = np.min(x_points) - x_pad
            x_max = np.max(x_points) + x_pad
        
        # Create fine x array for smooth interval zones
        x_fine = np.linspace(x_min, x_max, 1000)
        
        # ==================== PLOT ELDF CURVE ====================
        
        # Plot discrete ELDF points
        if eldf_vals is not None:
            ax1.plot(x_points, eldf_vals, 'o', color='blue', label='ELDF', markersize=4, alpha=0.7)
        
        # Plot smooth ELDF curve if available
        if smooth_x is not None and smooth_eldf is not None:
            ax1.plot(smooth_x, smooth_eldf, '-', color='blue', linewidth=2.5, alpha=0.9)
        
        # Plot WEDF if available
        if wedf_vals is not None:
            ax1.plot(x_points, wedf_vals, 's', color='lightblue', 
                    label='WEDF', markersize=3, alpha=0.6)
        
        # ==================== PLOT PDF CURVE ====================
        
        # Plot discrete PDF points
        if pdf_vals is not None:
            ax2.plot(x_points, pdf_vals, 'o', color='red', label='PDF', markersize=4, alpha=0.7)
            max_pdf = np.max(pdf_vals)
        else:
            max_pdf = 1.0
        
        # Plot smooth PDF curve if available
        if smooth_x is not None and smooth_pdf is not None:
            ax2.plot(smooth_x, smooth_pdf, '-', color='red', linewidth=2.5, alpha=0.9)
            max_pdf = max(max_pdf, np.max(smooth_pdf))
        
        # ==================== PLOT FILLED INTERVALS ====================
        
        # 1. Tolerance Interval (Z0L to Z0U) - Light Green
        if hasattr(self, 'intv') and self.intv and hasattr(self.intv, 'z0l') and hasattr(self.intv, 'z0u'):
            tolerance_mask = (x_fine >= self.intv.z0l) & (x_fine <= self.intv.z0u)
            
            if smooth_x is not None and smooth_eldf is not None:
                # Interpolate ELDF values for smooth filling
                tolerance_eldf = np.interp(x_fine[tolerance_mask], smooth_x, smooth_eldf)
                ax1.fill_between(x_fine[tolerance_mask], 0, tolerance_eldf, 
                               alpha=0.4, color='lightgreen', 
                               label=f'Tolerance Interval [Z0L: {self.intv.z0l:.3f}, Z0U: {self.intv.z0u:.3f}]')
            else:
                # Fallback to simple vertical fill
                ax1.axvspan(self.intv.z0l, self.intv.z0u, alpha=0.15, color='lightgreen',
                           label=f'Tolerance Interval [Z0L: {self.intv.z0l:.3f}, Z0U: {self.intv.z0u:.3f}]')
        
        # 2. Typical Data Interval (ZL to ZU) - Light Blue
        if hasattr(self, 'intv') and self.intv and hasattr(self.intv, 'zl') and hasattr(self.intv, 'zu'):
            typical_mask = (x_fine >= self.intv.zl) & (x_fine <= self.intv.zu)
            
            if smooth_x is not None and smooth_eldf is not None:
                # Interpolate ELDF values for smooth filling
                typical_eldf = np.interp(x_fine[typical_mask], smooth_x, smooth_eldf)
                ax1.fill_between(x_fine[typical_mask], 0, typical_eldf, 
                               alpha=0.1, color='blue',
                               label=f'Typical Data Interval [ZL: {self.intv.zl:.3f}, ZU: {self.intv.zu:.3f}]')
            else:
                # Fallback to simple vertical fill
                ax1.axvspan(self.intv.zl, self.intv.zu, alpha=0.15, color='lightblue',
                           label=f'Typical Data Interval [ZL: {self.intv.zl:.3f}, ZU: {self.intv.zu:.3f}]')
        
        # ==================== PLOT CRITICAL VERTICAL LINES ====================
        
        # ZL (lower datum) - Purple dashed
        if hasattr(self, 'intv') and self.intv and hasattr(self.intv, 'zl'):
            ax1.axvline(x=self.intv.zl, color='purple', linestyle='--', linewidth=2, 
                       alpha=0.8, label=f'ZL={self.intv.zl:.3f}')
        
        # Z0 (gnostic mode) - GREEN SOLID (most prominent)
        z0_value = None
        if hasattr(self, 'intv') and self.intv and hasattr(self.intv, 'z0'):
            z0_value = self.intv.z0
        elif hasattr(self, 'intv') and self.intv and hasattr(self.intv, 'z0_original'):
            z0_value = self.intv.z0_original
        elif hasattr(self.init_eldf, 'z0'):
            z0_value = self.init_eldf.z0
        
        if z0_value is not None:
            ax1.axvline(x=z0_value, color='magenta', linestyle='-.', linewidth=1, 
                       alpha=0.9, label=f'Z0={z0_value:.3f}')
        
        # ZU (upper datum) - Orange dashed  
        if hasattr(self, 'intv') and self.intv and hasattr(self.intv, 'zu'):
            ax1.axvline(x=self.intv.zu, color='orange', linestyle='--', linewidth=2, 
                       alpha=0.8, label=f'ZU={self.intv.zu:.3f}')
        
        # ==================== PLOT DATA BOUNDS ====================
        
        # Data bounds (DLB, DUB) - Solid lines
        if hasattr(self.init_eldf, 'DLB') and self.init_eldf.DLB is not None:
            ax1.axvline(x=self.init_eldf.DLB, color='green', linestyle='-', 
                       linewidth=1, alpha=0.6, label=f'DLB={self.init_eldf.DLB:.3f}')
        if hasattr(self.init_eldf, 'DUB') and self.init_eldf.DUB is not None:
            ax1.axvline(x=self.init_eldf.DUB, color='orange', linestyle='-', 
                       linewidth=1, alpha=0.6, label=f'DUB={self.init_eldf.DUB:.3f}')
        
        # Probable bounds (LB, UB) - Dashed lines
        if hasattr(self.init_eldf, 'LB') and self.init_eldf.LB is not None:
            ax1.axvline(x=self.init_eldf.LB, color='purple', linestyle='--', 
                       linewidth=1, alpha=0.6, label=f'LB={self.init_eldf.LB:.3f}')
        if hasattr(self.init_eldf, 'UB') and self.init_eldf.UB is not None:
            ax1.axvline(x=self.init_eldf.UB, color='brown', linestyle='--', 
                       linewidth=1, alpha=0.6, label=f'UB={self.init_eldf.UB:.3f}')
        
        # ==================== PLOT DATA POINTS (RUG) - NO LEGEND ====================
        
        # Add original data points as rug plot at bottom (no legend entry)
        data_y_pos = -0.05  # Position below x-axis
        ax1.scatter(x_points, [data_y_pos] * len(x_points), 
                   alpha=0.6, s=15, color='black', marker='|')
        
        # ==================== FORMATTING ====================
        
        # Set axis labels and limits
        ax1.set_xlabel('Data Values', fontsize=12, fontweight='bold')
        ax1.set_ylabel('ELDF Value', fontsize=12, fontweight='bold', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_ylim(-0.1, 1.05)  # Extended to show rug plot
        ax1.set_xlim(x_min, x_max)
        
        ax2.set_ylabel('PDF Value', fontsize=12, fontweight='bold', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(0, max_pdf * 1.1)
        ax2.set_xlim(x_min, x_max)
        
        # Grid
        ax1.grid(True, alpha=0.3)
        
        # Simplified title with Z0 information
        df_type = 'ELDF'  # Default to ELDF, but detect actual type
        if hasattr(self.init_eldf, '__class__'):
            class_name = self.init_eldf.__class__.__name__
            if 'EGDF' in class_name:
                df_type = 'EGDF'
        
        # Build title with Z0 value
        title_text = f'{df_type} Interval Analysis'
        if z0_value is not None:
            title_text += f' (Z0 = {z0_value:.3f})'
        
        plt.title(title_text, fontsize=14, fontweight='bold')
        
        # Combined legend - organized by category
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        
        # Filter and organize legend entries
        interval_entries = []
        critical_entries = []
        bound_entries = []
        curve_entries = []
        
        # Process ax1 entries
        for line, label in zip(lines1, labels1):
            if 'Interval' in label:
                interval_entries.append((line, label))
            elif any(marker in label for marker in ['ZL=', 'Z0=', 'ZU=']):
                critical_entries.append((line, label))
            elif any(bound in label for bound in ['LB=', 'UB=', 'DLB=', 'DUB=']):
                bound_entries.append((line, label))
            elif any(curve in label for curve in ['ELDF', 'WEDF']):
                curve_entries.append((line, label))
        
        # Process ax2 entries
        for line, label in zip(lines2, labels2):
            if 'PDF' in label:
                curve_entries.append((line, label))
        
        # Combine in logical order: intervals, critical points, curves, bounds
        all_entries = interval_entries + critical_entries + curve_entries + bound_entries
        
        if all_entries:
            all_lines, all_labels = zip(*all_entries)
            ax1.legend(all_lines, all_labels, loc='upper left', fontsize=10, 
                      bbox_to_anchor=(0.02, 0.98))
        
        plt.tight_layout()
        plt.show()
        
        # Print summary information with Z0
        if self.verbose:
            print(f"\n{df_type} Interval Analysis Plot Summary:")
            if hasattr(self, 'intv') and self.intv:
                if z0_value is not None:
                    print(f"  Z0 (Gnostic Mode): {z0_value:.4f}")
                if hasattr(self.intv, 'z0l') and hasattr(self.intv, 'z0u'):
                    print(f"  Tolerance interval: [{self.intv.z0l:.4f}, {self.intv.z0u:.4f}] (width: {self.intv.tolerance_interval:.4f})")
                if hasattr(self.intv, 'zl') and hasattr(self.intv, 'zu'):
                    print(f"  Typical data interval: [{self.intv.zl:.4f}, {self.intv.zu:.4f}] (width: {self.intv.typical_data_interval:.4f})")
                
                # Data coverage analysis
                if hasattr(self.intv, 'z0l') and hasattr(self.intv, 'z0u'):
                    data_in_tolerance = np.sum((x_points >= self.intv.z0l) & (x_points <= self.intv.z0u))
                    print(f"  Data coverage - Tolerance: {data_in_tolerance}/{len(x_points)} ({data_in_tolerance/len(x_points):.1%})")
                if hasattr(self.intv, 'zl') and hasattr(self.intv, 'zu'):
                    data_in_typical = np.sum((x_points >= self.intv.zl) & (x_points <= self.intv.zu))
                    print(f"  Data coverage - Typical: {data_in_typical}/{len(x_points)} ({data_in_typical/len(x_points):.1%})")
            
            print(f"  Total data points: {len(x_points)}")
            print(f"  Data range: [{np.min(x_points):.4f}, {np.max(x_points):.4f}]")