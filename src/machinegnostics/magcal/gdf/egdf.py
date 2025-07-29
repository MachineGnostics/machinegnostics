"""
EGDF - Estimating Global Distribution Function.

Author: Nirmal Parmar
Machine Gnostics
"""

import numpy as np
from machinegnostics.magcal.gdf.base_egdf import BaseEGDF

class EGDF(BaseEGDF):
    """
    EGDF - A class for estimating the global distribution function.
    """
    def __init__(self,
                 data: np.ndarray,
                 DLB: float = None,
                 DUB: float = None,
                 LB: float = None,
                 UB: float = None,
                 S = 'auto',
                 tolerance: float = 1e-3,
                 data_form: str = 'a',
                 n_points: int = 100,
                 homogeneous: bool = True,
                 catch: bool = True,
                 weights: np.ndarray = None):
        
        """
        Initialize the EGDF class.

        Parameters:
        data (np.ndarray): Input data for the EGDF.
        DLB (float): Lower bound for the data.
        DUB (float): Upper bound for the data.
        LB (float): Lower (Probable) Bound.
        UB (float): Upper (Probable) Bound.
        S (float): Scale parameter.
        tolerance (float): Tolerance for convergence.
        data_form (str): Form of the data ('a' for additive, 'm' for multiplicative).
        n_points (int): Number of points in the distribution function.
        homogeneous (bool): Whether given data is homogeneous, True by default. if False, in that case data will be homogenized.
        catch (bool): To catch calculated values or not, True by default.
        weights (np.ndarray): Priory Weights for the data points.
        data_pad (float): Padding for the data range.
        """
        
        self.data = data
        self.DLB = DLB
        self.DUB = DUB
        self.LB = LB
        self.UB = UB
        self.S = S
        self.tolerance = tolerance
        self.data_form = data_form
        self.n_points = n_points
        self.homogeneous = homogeneous
        self.catch = catch
        self.weights = weights if weights is not None else np.ones_like(data)
        self.params = {}

    #1
    def fit(self):
        self._fit()

    #2
    def plot(self, plot_smooth: bool = True):
        self._plot(plot_smooth=plot_smooth)

    def marginal_analysis(self):
        """
        Perform marginal analysis on the EGDF.

        This method can be overridden in subclasses to provide custom marginal analysis logic.
        """
        # Default implementation does nothing
        pass
        
