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
    def __init__(self,
            data: np.ndarray,
            early_stopping_steps: int = 10,
            cluster_threshold: float = 0.05,
            estimate_cluster_bounds: bool = True,
            get_clusters: bool = True,
            n_points_per_direction: int = 1000,
            dense_zone_fraction: float = 0.4,
            dense_points_fraction: float = 0.7,
            convergence_window: int = 15,
            convergence_threshold: float = 1e-7,
            min_search_points: int = 30,
            boundary_margin_factor: float = 0.001,
            extrema_search_tolerance: float = 1e-6,
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
            flush: bool = True):
        super().__init__(data=data,
                         early_stopping_steps=early_stopping_steps,
                         cluster_threshold=cluster_threshold,
                         get_clusters=get_clusters,
                        n_points_per_direction=n_points_per_direction,
                        dense_zone_fraction=dense_zone_fraction,
                        dense_points_fraction=dense_points_fraction,
                        convergence_window=convergence_window,
                        convergence_threshold=convergence_threshold,
                        min_search_points=min_search_points,
                        boundary_margin_factor=boundary_margin_factor,
                        extrema_search_tolerance=extrema_search_tolerance,
                        estimate_cluster_bounds=estimate_cluster_bounds,
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
        self.early_stopping_steps = early_stopping_steps
        self.cluster_threshold = cluster_threshold
        self.get_clusters = get_clusters
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
        self.params = {}

        # fit status
        self._fitted = False

        # validation inputs

    def _fit_eldf_intv(self):
        # fit ELDF
        # extract z0, and bounds
        self._get_init_eldf_fit()

        # homogeneity check, pick counts, cluster bounds
        self.is_homogeneous = self._is_homogeneous()

        if self.is_homogeneous:
            if self.verbose:
                print("Data is homogeneous. Using homogeneous data for interval analysis.")
        else:
            if self.verbose:
                print("Data is heterogeneous. Need to estimate cluster bounds to find main cluster.")
        
        # h check
            if self.h == False and self.estimate_cluster_bounds == False and self.get_clusters == True:
                warnings.warn("Data is heterogeneous but estimate_cluster_bounds is False. "
                            "Consider setting 'estimate_cluster_bounds=True' and 'get_clusters=True' to find main cluster bounds and main cluster.")

        # get main cluster and other clusters
        self.lower_cluster, self.main_cluster, self.upper_cluster = self._get_cluster()
        
        # if data is not homogeneous, user option
        # get the main cluster and the cluster bounds
        if not self.homogeneous and self.get_clusters and self.main_cluster is not None and len(self.main_cluster) > 4: # + user choice
            self._get_main_init_eldf(self.main_cluster)
        
        if len(self.main_cluster) < 4:
            warnings.warn("Main cluster has fewer than 4 data points. ELDF fitting and cluster analysis may be unreliable. Please check the data.")
            # exit without fitting
            return None
        
        if self.verbose:
                print("Initiating EGDF Interval Analysis...")
                
        # intv analysis
        # get extended ELDF with a new datum

        # get new z0 with a new datum


        pass

    def _plot_eldf_intv(self):
        pass

    def _get_main_init_eldf(self, cluster: np.ndarray):
        if self.verbose:
            print("Fitting initial ELDF to the main cluster...")

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
        if self.verbose:
            print("Initial ELDF fit to the main cluster completed.")
