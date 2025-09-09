'''
QLDF Quantifying Local Distribution Function (QLDF)

Public class
Author: Nirmal Parmar
Machine Gnostics
'''

import numpy as np
from machinegnostics.magcal.gdf.base_qldf import BaseQLDF

class QLDF(BaseQLDF):

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
        """Initialize QLDF with parameters and verbosity."""
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
                         flush=flush)
    def fit(self, plot: bool = False):
        """Public method to fit QLDF and optionally plot results."""
        self._fit_qldf(plot=plot)

    def plot(self, plot_smooth: bool = True, plot: str = 'both', bounds: bool = True, extra_df: bool = True, figsize: tuple = (12, 8)):
        """Public method to plot QLDF and/or PDF results."""
        self._plot(plot_smooth=plot_smooth, plot=plot, bounds=bounds, extra_df=extra_df, figsize=figsize)

    def results(self) -> dict:
        """Public method to get fitting results."""
        return self._get_results()