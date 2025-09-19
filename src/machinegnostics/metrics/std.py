'''
Gnostic standard deviation of given sample

method: std()

Authors: Nirmal Parmar
Machine Gnostics
'''

import numpy as np
from machinegnostics.metrics.mean import mean
from machinegnostics.metrics.variance import variance

def std(data: np.ndarray,
        case: str = 'i',
        S: float = 1,
        z0_optimize: bool = True,
        data_form: str = 'a',
        tolerance: float = 1e-6) -> tuple:
    """
    Calculate the standard deviation of the given data.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    case : str, optional
        Case for irrelevance calculation ('i' or 'j'). Default is 'i'. 
        'i' for estimating variance, 'j' for quantifying variance.
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
    tuple
        Lower and upper bounds of the standard deviation.

    Example:
    --------
    >>> import machinegnostics as mg
    >>> import numpy as np
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> mg.std(data)
    (2.9403976979154143, 3.0599336862362043)
    """
    # Validate input
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    
    if data.ndim != 1:
        raise ValueError("Input array must be one-dimensional.")
    
    # mean
    m = mean(data, S=S, z0_optimize=z0_optimize, data_form=data_form, tolerance=tolerance)

    # variance
    v = np.abs(variance(data, case=case, S=S, z0_optimize=z0_optimize, data_form=data_form, tolerance=tolerance))

    # std
    if case.lower() == 'i':
        std_value_ub = m * ((1 + np.sqrt(v)) / ( 1 - np.sqrt(v)))**(S/2)
        std_value_lb = m * ((1 - np.sqrt(v)) / ( 1 + np.sqrt(v)))**(S/2)

    elif case.lower() == 'j':
        std_value_ub = m * ((1 + np.sqrt(v)) / ( 1 - np.sqrt(v)))**(S)
        std_value_lb = m * ((1 - np.sqrt(v)) / ( 1 + np.sqrt(v)))**(S)

    else:
        raise ValueError("case must be either 'i' or 'j'. i for estimating variance, j for quantifying variance.")


    return float(std_value_lb), float(std_value_ub)