'''
Gnostic standard deviation of given sample

method: std()

Authors: Nirmal Parmar
Machine Gnostics
'''

import logging
from machinegnostics.magcal.util.logging import get_logger
import numpy as np
from machinegnostics.metrics.mean import mean
from machinegnostics.metrics.variance import variance
from machinegnostics.magcal import EGDF, ELDF
from machinegnostics.magcal import DataConversion

def std(data: np.ndarray,
        case: str = 'i',
        S: float = 'auto',
        z0_optimize: bool = True,
        data_form: str = 'a',
        tolerance: float = 1e-6,
        verbose: bool = False) -> tuple:
    """
    Calculate the standard deviation of the given data.

    The Gnostic standard deviation metric is based on the principles of gnostic theory, which
    provides robust estimates of data relationships. This metric leverages the concepts
    of estimating irrelevances and fidelities, and quantifying irrelevances and fidelities, which are robust measures of data uncertainty. These irrelevances are aggregated differently.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    case : str, optional
        Case for irrelevance calculation ('i' or 'j'). Default is 'i'. 
        'i' for estimating variance, 'j' for quantifying variance.
    S: Scaling parameter for EGDF. Default is 'auto' to optimize using EGDF.
            Suggested range is [0.01, 10].
    z0_optimize : bool, optional
        Whether to optimize z0 in ELDF. Default is True.    
    data_form : str, optional
        Data form for ELDF. Default is 'a'. 'a' for additive, 'm' for multiplicative.
    tolerance : float, optional
        Tolerance for ELDF fitting. Default is 1e-6.
    verbose : bool, optional
        If True, enables detailed logging for debugging purposes. Default is False.

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
    logger = get_logger('std', level=logging.WARNING if not verbose else logging.INFO)
    logger.info("Calculating standard deviation...")

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
    
    # mean
    logger.info("Calculating mean...")
    m = mean(data, S=S, z0_optimize=z0_optimize, data_form=data_form, tolerance=tolerance)

    # variance
    logger.info("Calculating variance...")
    v = np.abs(variance(data, case=case, S=S, z0_optimize=z0_optimize, data_form=data_form, tolerance=tolerance))

    # EGDF fitting 
    logger.info("Optimizing S using EGDF...")
    egdf = EGDF(z0_optimize=z0_optimize, data_form=data_form, tolerance=tolerance, verbose=verbose, S=S)
    egdf.fit(data=data, plot=False)
    S = egdf.S_opt

    # data domain conversion
    logger.info("Converting data domain...")
    if egdf.data_form == 'm':
        m_z = DataConversion._convert_mz(m, egdf.DLB, egdf.DUB)
    else:
        m_z = DataConversion._convert_az(m, egdf.DLB, egdf.DUB)
    # to infinite domain
    mzi = DataConversion._convert_fininf(m_z, egdf.LB, egdf.UB)

    # std
    if case.lower() == 'i':
        # safe check
        if 1 - np.sqrt(v) <= 0:
            logger.warning("Encountered negative sqrt value!")
            return 0, 0
        logger.info("Calculating standard deviation the estimating geometry...")
        std_value_ub = mzi * ((1 + np.sqrt(v)) / ( 1 - np.sqrt(v) + 1e-6))**(S/2)
        std_value_lb = mzi * ((1 - np.sqrt(v)) / ( 1 + np.sqrt(v) + 1e-6))**(S/2)

    elif case.lower() == 'j':
        logger.info("Calculating standard deviation the quantifying geometry...")

        std_value_ub = mzi * ((np.sqrt(v)) + ( 1 + np.sqrt(v)))**(S/2)
        std_value_lb = mzi * ((np.sqrt(v)) + ( 1 - np.sqrt(v)))**(S/2)

    else:
        raise ValueError("case must be either 'i' or 'j'. i for estimating variance, j for quantifying variance.")


        
    # back to data domain
    logger.info("Converting standard deviation back to original data domain...")
    if egdf.data_form == 'm':
        std_value_ub = DataConversion._convert_inffin(std_value_ub, egdf.LB, egdf.UB)
        std_value_ub = DataConversion._convert_zm(std_value_ub, egdf.DLB, egdf.DUB)

        std_value_lb = DataConversion._convert_inffin(std_value_lb, egdf.LB, egdf.UB)
        std_value_lb = DataConversion._convert_zm(std_value_lb, egdf.DLB, egdf.DUB)
    else:
        std_value_ub = DataConversion._convert_inffin(std_value_ub, egdf.LB, egdf.UB)
        std_value_ub = DataConversion._convert_za(std_value_ub, egdf.DLB, egdf.DUB)

        std_value_lb = DataConversion._convert_inffin(std_value_lb, egdf.LB, egdf.UB)
        std_value_lb = DataConversion._convert_za(std_value_lb, egdf.DLB, egdf.DUB)
    
    logger.info("Gnostic standard deviation calculation completed.")

    # delta
    delta = (std_value_ub - std_value_lb)

    return m - delta/2, m + delta/2