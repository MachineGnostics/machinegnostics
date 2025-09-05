'''
ELDF Interval Analysis Module

Estimating Local Marginal Analysis

Author: Nirmal Parmar
Machine Gnostics
'''

import numpy as np
from machinegnostics.magcal.gdf.base_el_intv import BaseIntervalAnalysisELDF

class IntervalAnalysisELDF(BaseIntervalAnalysisELDF):
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
            flush=flush,
            early_stopping_steps=early_stopping_steps,
            cluster_threshold=cluster_threshold,
            estimate_cluster_bounds=estimate_cluster_bounds,
            get_clusters=get_clusters,
            n_points_per_direction=n_points_per_direction,
            dense_zone_fraction=dense_zone_fraction,
            dense_points_fraction=dense_points_fraction,
            convergence_window=convergence_window,
            convergence_threshold=convergence_threshold,
            min_search_points=min_search_points,
            boundary_margin_factor=boundary_margin_factor,
            extrema_search_tolerance=extrema_search_tolerance)
        
    def fit(self, plot: bool = False):
        self._fit_eldf_intv(plot=plot)


    def plot(self, figsize=(12, 8)):
        """
        Plot the results of the ELDF interval analysis.
        """
        self._plot_eldf_intv(figsize=figsize)

    def get_intervals(self, decimals: int = 4):
        """
        Get the computed intervals from the ELDF interval analysis.

        Returns:
            dict: A dictionary containing the computed intervals.
        """
        return self.intv.get_intervals(decimals=decimals)
