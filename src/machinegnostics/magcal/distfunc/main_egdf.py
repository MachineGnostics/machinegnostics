import numpy as np

from machinegnostics.magcal.distfunc.base_egdf import BaseEGDF

class EGDF(BaseEGDF):
    """
    EGDF - Estimating Global Distribution Function.
    This class transforms data into a standard domain using the EGDF method.
    """

    def __init__(self,
                 data,
                 DLB: float = None,
                 DUB: float = None,
                 LSB: float = None,
                 USB: float = None,
                 LB: float = None,
                 UB: float = None,
                 S=1.0,
                 tolerance: float = 1e-5,
                 data_form: str = 'a',
                 n_points: int = 100,
                 homogeneous: bool = True,
                 catch: bool = True,
                 weights: np.ndarray = None):
        super().__init__(data=data, DLB=DLB, DUB=DUB, LSB=LSB, USB=USB, LB=LB, UB=UB, S=S,
                         tolerance=tolerance, data_form=data_form, n_points=n_points,
                         homogeneous=homogeneous, catch=catch, weights=weights)

    def fit(self):
        """
        Fit the EGDF model to the data.
        This method applies the transformation and prepares the data for further analysis.
        """
        self._fit()
        # Additional fitting logic can be added here if needed
        return self