'''
Gnostic Variance of given sample data

method: variance()

Authors: Nirmal Parmar
Machine Gnostics
'''
import numpy as np
from machinegnostics.metrics.mean import mean
from machinegnostics.magcal import ELDF

def variance(data: np.ndarray,
             case: str = 'i', 
             S: float = 1, 
             z0_optimize: bool = True, 
             data_form: str = 'a',
             tolerance: float = 1e-6) -> float:
    """
    Calculate the gnostic variance of the given data.

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
    float
        Gnostic variance of the data.

    Example:
    --------
    >>> import machinegnostics as mg
    >>> import numpy as np
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> mg.variance(data)
    2.5
    """
    # Validate input
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array.")

    # Compute eldf
    eldf = ELDF(homogeneous=True, S=S, z0_optimize=z0_optimize, tolerance=tolerance, data_form=data_form, wedf=False)
    eldf.fit(data, plot=False)

    # point specific GC
    gc, q, q1 = eldf._calculate_gcq_at_given_zi(eldf.z0)

    # Compute irrelevance 
    if case.lower() == 'i':
        hc = np.mean(gc._hi(q, q1))
    elif case.lower() == 'j':
        hc = np.mean(gc._hj(q, q1))
    else:
        raise ValueError("case must be either 'i' or 'j'. i for estimating variance, j for quantifying variance.")

    return float(hc)

