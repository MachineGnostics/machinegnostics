'''
To calculate gnostic median

Author: Nirmal Parmar
Machine Gnostics
'''

import numpy as np
from machinegnostics.magcal import EGDF

def median(data,
            S: float = 1, 
            z0_optimize: bool = True, 
            data_form: str = 'a',
            tolerance: float = 1e-6):
    """
    Calculate the median of a dataset.

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
        Median of the data.

    Example:
    --------
    >>> import machinegnostics as mg
    >>> import numpy as np
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> mg.median(data)
    3.0

    """
    data = np.asarray(data)
    n = len(data)
    
    # Validate input
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Input data must be a one-dimensional array.")
    if len(data) == 0:
        raise ValueError("Input data array is empty.")
    
    # egdf
    egdf = EGDF(homogeneous=True, S=S, z0_optimize=z0_optimize, tolerance=tolerance, data_form=data_form)
    egdf.fit(data, plot=False)
    median_value = egdf.z0
    
    return median_value