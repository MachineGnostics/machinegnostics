"""
EGDF - Estimating Global Distribution Function.

Author: Nirmal Parmar
Machine Gnostics
"""

import numpy as np
from machinegnostics.magcal.gdf.base_egdf import BaseEGDF

class EGDF(BaseEGDF):
    """
    EGDF - Estimating Global Distribution Function.
    
    A comprehensive class for estimating and analyzing global distribution functions (EGDF).
    This class provides methods to fit distribution functions, visualize results, and perform marginal analysis on data with optional bounds and weighting.

    The EGDF class supports both additive and multiplicative data forms and can handle bounded and
    unbounded data distributions. It provides automatic parameter estimation and flexible visualization options for distribution analysis.
    
    Attributes:
        data (np.ndarray): The input dataset used for distribution estimation.
        DLB (float): Data Lower Bound - absolute minimum value the data can take.
        DUB (float): Data Upper Bound - absolute maximum value the data can take.
        LB (float): Lower Probable Bound - practical lower limit for the distribution.
        UB (float): Upper Probable Bound - practical upper limit for the distribution.
        S (float or str): Scale parameter for the distribution. Set to 'auto' for automatic estimation.
        data_form (str): Form of the data processing:
            - 'a': Additive form (default)
            - 'm': Multiplicative form
        n_points (int): Number of points to generate in the distribution function (default: 100).
        catch (bool): Whether to store intermediate calculated values (default: True).
        weights (np.ndarray): Prior weights for data points. If None, uniform weights are used.
        params (dict): Dictionary storing fitted parameters and results.
    
    Methods:
        fit(): Fit the global distribution function to the data.
        plot(plot_smooth=True, plot='gdf', bounds=False): Visualize the fitted distribution.
        marginal_analysis(): Perform marginal analysis on the distribution.
    
    Examples:
        Basic usage with default parameters:
        >>> import numpy as np
        >>> from machinegnostics.magcal import EGDF
        >>> # Stack Loss example data
        >>> data = [7, 8, 8, 8, 9, 11, 12, 13, 14, 14, 15, 15, 15, 18, 18, 19, 20, 28, 37, 37, 42]
        >>> data = np.array(data)
        >>> egdf = EGDF(data)
        >>> egdf.fit()
        >>> egdf.plot()
        
        Usage with custom bounds and weights:
        >>> data = np.random.exponential(2, 500)
        >>> weights = np.random.uniform(0.5, 1.5, 500)
        >>> egdf = EGDF(data, DLB=0, DUB=20, LB=0.1, UB=15, weights=weights)
        >>> egdf.fit()
        >>> egdf.plot(bounds=True)
        
        Multiplicative form with custom scale:
        >>> data = np.random.lognormal(0, 0.5, 800)
        >>> egdf = EGDF(data, data_form='m', S=2.0, n_points=200)
        >>> egdf.fit()
        >>> egdf.marginal_analysis()
    
    Notes:
        - Bounds (DLB, DUB, LB, UB) are optional but can improve estimation accuracy
        - When S='auto', the scale parameter is automatically estimated from the data
        - The weights array must have the same length as the data array
        - Setting catch=False can save memory for large datasets but prevents access to intermediate results
    """
    def __init__(self,
                 data: np.ndarray,
                 DLB: float = None,
                 DUB: float = None,
                 LB: float = None,
                 UB: float = None,
                 S = 'auto',
                 data_form: str = 'a',
                 n_points: int = 100,
                 catch: bool = True,
                 weights: np.ndarray = None,
                 wedf: bool = True):
        
        """
        Initialize the EGDF class.

        Parameters:
        data (np.ndarray): Input data for the EGDF.
        DLB (float): Lower bound for the data.
        DUB (float): Upper bound for the data.
        LB (float): Lower (Probable) Bound.
        UB (float): Upper (Probable) Bound.
        S (float): Scale parameter.
        data_form (str): Form of the data ('a' for additive, 'm' for multiplicative).
        n_points (int): Number of points in the distribution function.
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
        self.tolerance = 1e-3
        self.data_form = data_form
        self.n_points = n_points
        self.homogeneous = True
        self.catch = catch
        self.weights = weights if weights is not None else np.ones_like(data)
        self.wedf = wedf
        self.params = {}

    #1
    def fit(self):
        self._fit()

    #2  
    def plot(self, 
             plot_smooth: bool = True, 
             plot: str ='gdf', 
             bounds: bool = False,
             extra_df: bool = True):
        self._plot(plot_smooth=plot_smooth, 
                   plot=plot, 
                   bounds=bounds, 
                   extra_df=extra_df)

    #3
    def marginal_analysis(self):
        """
        Perform marginal analysis on the EGDF.

        This method can be overridden in subclasses to provide custom marginal analysis logic.
        """
        # Default implementation does nothing
        pass
        
