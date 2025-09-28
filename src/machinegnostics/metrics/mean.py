'''
calculate gnostic mean of given sample

method: mean()
LEL local estimate of location

Authors: Nirmal Parmar
Machine Gnostics
'''
import logging
from machinegnostics.magcal.util.logging import get_logger
import numpy as np
from machinegnostics.magcal import ELDF, QLDF
from typing import Union

def mean(data: np.ndarray, 
         S: Union[float, str] = 1, 
         case: str = 'i',
         z0_optimize: bool = True, 
         data_form: str = 'a',
         tolerance: float = 1e-6,
         verbose: bool = False) -> float:
    """
    Calculate the gnostic mean of the given data.

    The Gnostic mean metric is based on the principles of gnostic theory, which
    provides robust estimates of data relationships. This metric leverages the concepts
    of estimating irrelevances and fidelities, and quantifying irrelevances and fidelities, which are robust measures of data uncertainty. These irrelevances are aggregated differently.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    S : float, optional
        Scaling parameter for ELDF. Default is 1.
    case : str, optional
        Case for irrelevance calculation ('i' or 'j'). Default is 'i'. 
        'i' for estimating irrelevance, 'j' for quantifying irrelevance.
    z0_optimize : bool, optional
        Whether to optimize z0 in ELDF. Default is True.
    data_form : str, optional
        Data form for ELDF. Default is 'a'. 'a' for additive, 'm' for multiplicative.
    tolerance : float, optional
        Tolerance for ELDF fitting. Default is 1e-6.
    verbose : bool, optional
        If True, enables detailed logging for debugging purposes.

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
    """
    logger = get_logger('mean', level=logging.WARNING if not verbose else logging.INFO)
    logger.info("Calculating gnostic mean...")

    # flatten data
    data = np.asarray(data).flatten()
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

    # arg validation
    if isinstance(S, str):
        if S != 'auto':
            logger.error("S must be a float or 'auto'.")
            raise ValueError("S must be a float or 'auto'.")
    elif not isinstance(S, (int, float)):
        logger.error("S must be a float or 'auto'.")
        raise TypeError("S must be a float or 'auto'.")
    # S proper value [0,2] suggested
    if isinstance(S, (int)):
        if S < 0 or S > 2:
            logger.warning("S must be in the range [0, 2].")
    # Check for valid data_form
    if data_form not in ['a', 'm']:
        logger.error("data_form must be 'a' for additive or 'm' for multiplicative.")
        raise ValueError("data_form must be 'a' for additive or 'm' for multiplicative.")
    
    if case == 'i':
        logger.info("Using estimating geometry for mean calculation.")
        # Compute eldf
        eldf = ELDF(homogeneous=True, S=S, z0_optimize=z0_optimize, tolerance=tolerance, data_form=data_form, wedf=False)
        eldf.fit(data, plot=False)
        mean_value = eldf.z0
    else:
        logger.info("Using quantifying geometry for mean calculation.")
        # Compute qldf
        qldf = QLDF(homogeneous=True, S=S, z0_optimize=z0_optimize, tolerance=tolerance, data_form=data_form, wedf=False)
        qldf.fit(data, plot=False)
        mean_value = qldf.z0
    logger.info(f"Gnostic mean calculated.")

    return float(mean_value)