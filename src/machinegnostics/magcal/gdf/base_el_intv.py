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

        # verbose message
        if self.verbose:
            print("ELDF Interval Analysis completed.")

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

    def _plot_eldf_intv(self, figsize=(12, 8)):
        """
        Generate diagnostic plots for interval analysis results.
        
        Developer Notes:
        ----------------
        Creates visualization of both the interval analysis results and the
        underlying initial ELDF fit. Useful for debugging and result interpretation.
        
        Raises:
        -------
        RuntimeError : If called before fitting is complete
        """
        if not self._fitted:
            raise RuntimeError(
                "Cannot generate plots: Interval analysis not yet fitted. "
                "Please call the 'fit' method before attempting to plot results."
            )
        self.intv.plot(figsize=figsize)
        self.init_eldf.plot(figsize=figsize)

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