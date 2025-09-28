'''
Gnostic Variance of given sample data

method: variance()

Authors: Nirmal Parmar
Machine Gnostics
'''
import numpy as np
import logging
from machinegnostics.metrics.mean import mean
from machinegnostics.magcal import ELDF, QLDF
from machinegnostics.magcal.util.logging import get_logger

def variance(data: np.ndarray,
             case: str = 'i', 
             S: float = 1, 
             z0_optimize: bool = True, 
             data_form: str = 'a',
             tolerance: float = 1e-6,
             verbose: bool = False) -> float:
    """
    Calculate the gnostic variance of the given data.

    The Gnostic variance metric is based on the principles of gnostic theory, which
    provides robust estimates of data relationships. This metric leverages the concepts
    of estimating irrelevances and quantifying irrelevances, which are robust measures
    of data uncertainty. These irrelevances are aggregated differently.

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
    0.002685330177795109
    """
    logger = get_logger('variance', level=logging.WARNING if not verbose else logging.INFO)

    logger.info("Calculating gnostic variance...")
    # Validate input
    if not isinstance(data, np.ndarray):
        logger.error("Input must be a numpy array.")
        raise TypeError("Input must be a numpy array.")
    if data.ndim != 1:
        logger.error("Input data must be a one-dimensional array.")
        raise ValueError("Input data must be a one-dimensional array.")
    if len(data) == 0:
        logger.error("Input data array is empty.")
        raise ValueError("Input data array is empty.")
    if np.any(np.isnan(data)):
        logger.error("Input data contains NaN values.")
        raise ValueError("Input data contains NaN values.")
    if np.any(np.isinf(data)):
        logger.error("Input data contains Inf values.")
        raise ValueError("Input data contains Inf values.")
    # Check for valid case
    if case not in ['i', 'j']:
        logger.error("Case must be 'i' for estimating variance or 'j' for quantifying variance.")
        raise ValueError("Case must be 'i' for estimating variance or 'j' for quantifying variance.")
    
    if case == 'i':
        logger.info("Using ELDF for variance calculation...")
        # Compute eldf
        eldf = ELDF(homogeneous=True, S=S, z0_optimize=z0_optimize, tolerance=tolerance, data_form=data_form, wedf=False, flush=False)
        eldf.fit(data, plot=False)
        hi = eldf.hi
        hc = np.mean(hi**2)
    
    if case == 'j':
        logger.info("Using QLDF for variance calculation...")
        # Compute qldf
        qldf = QLDF(homogeneous=True, S=S, z0_optimize=z0_optimize, tolerance=tolerance, data_form=data_form, wedf=False, flush=False)
        qldf.fit(data)
        hj = qldf.hj
        hc = np.mean(hj**2)

    logger.info(f"Gnostic variance calculated.")

    return float(hc)