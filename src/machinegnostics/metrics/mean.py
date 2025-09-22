'''
calculate gnostic mean of given sample

method: mean()

Authors: Nirmal Parmar
Machine Gnostics
'''

import numpy as np
from machinegnostics.magcal import ELDF

def mean(data: np.ndarray, 
         S: float = 1, 
         z0_optimize: bool = True, 
         data_form: str = 'a',
         tolerance: float = 1e-6) -> float:
    """
    Calculate the gnostic mean of the given data.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    S : float, optional
        Scaling parameter for ELDF. Default is 1.
    z0_optimize : bool, optional
        Whether to optimize z0 in ELDF. Default is True.
    data_form : str, optional
        Data form for ELDF. Default is 'a'. 'a' for additive, 'm' for multiplicative.
    tolerance : float, optional
        Tolerance for ELDF fitting. Default is 1e-6.

    Returns:
    --------
    float
        Gnostic mean of the data.

    Example:
    --------
    >>> import machinegnostics as mg
    >>> import numpy as np
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> mg.mean(data)
    3.0
    """
    # Validate input
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Input data must be a one-dimensional array.")
    if len(data) == 0:
        raise ValueError("Input data array is empty.")
    
    # Compute eldf
    eldf = ELDF(homogeneous=True, S=S, z0_optimize=z0_optimize, tolerance=tolerance, data_form=data_form, wedf=False)
    eldf.fit(data, plot=False)
    mean_value = eldf.z0

    return float(mean_value)