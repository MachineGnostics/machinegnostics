'''
ManGo - Machine Gnostics Library
Copyright (C) 2025  ManGo Team

Author: Nirmal Parmar
'''

from machinegnostics.magcal.util.logging import get_logger
import logging
import numpy as np
from machinegnostics.magcal.characteristics import GnosticsCharacteristics

def hc(y_true: np.ndarray, y_pred: np.ndarray, case: str = 'i', verbose: bool = False) -> float:
    """
    Calculate the Gnostic Characteristics (Hc) metric of the data sample.

    i  - for estimating gnostic relevance
        For case 'i': Range [0, 1]. Close to one indicates less relevance
    
    j  - for estimating gnostic irrelevance
        For case 'j': Range [0, ∞) (measures strength of relationship). Close to 1 indicates less irrelevance.

    The HC metric is used to evaluate the performance of a model by comparing
    the predicted values with the true values's relevance or irrelevance.
    It is calculated as the sum of the characteristics of the model. For standard comparison, irrelevances are calculated with S=1.

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.
    case : str, optional
        Case to be used for calculation. Options are 'i' or 'j'. Default is 'i'.
    verbose : bool, optional
        If True, enables detailed logging for debugging purposes. Default is False.

    Returns
    -------
    float
        The calculated HC value.

    Example
    -------
    >>> y_true = [1, 2, 3]
    >>> y_pred = [1, 2, 3]
    >>> from mango.metrics import hc
    >>> hc_value = hc(y_true, y_pred, case='i')
    >>> print(hc_value)
    """
    logger = get_logger('Hc', level=logging.WARNING if not verbose else logging.INFO)
    logger.info("Starting HC calculation.")

    # Validate input types
    if not isinstance(y_true, (list, tuple, np.ndarray)):
        logger.error("Invalid input type for y_true.")
        raise TypeError("y_true must be array-like (list, tuple, or numpy array).")
    if not isinstance(y_pred, (list, tuple, np.ndarray)):
        logger.error("Invalid input type for y_pred.")
        raise TypeError("y_pred must be array-like (list, tuple, or numpy array).")
    if case not in ['i', 'j']:
        logger.error("Invalid case value.")
        raise ValueError("case must be either 'i' or 'j'.")
    # Validate input dimensions
    if np.ndim(y_true) > 1:
        logger.error("y_true must be a 1D array.")
        raise ValueError("y_true must be a 1D array.")
    if np.ndim(y_pred) > 1:
        logger.error("y_pred must be a 1D array.")
        raise ValueError("y_pred must be a 1D array.")
    # Check for empty arrays
    if len(y_true) == 0:
        logger.error("y_true and y_pred must not be empty.")
        raise ValueError("y_true and y_pred must not be empty.")
    if len(y_pred) == 0:
        logger.error("y_true and y_pred must not be empty.")
        raise ValueError("y_true and y_pred must not be empty.")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        logger.error("y_true and y_pred must not contain NaN values.")
        raise ValueError("y_true and y_pred must not contain NaN values.")
    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        logger.error("y_true and y_pred must not contain Inf values.")
        raise ValueError("y_true and y_pred must not contain Inf values.")  
    # Convert to numpy arrays and flatten if necessary
    if isinstance(y_true, (list, tuple)):
        y_true = np.array(y_true)
    if isinstance(y_pred, (list, tuple)):
        y_pred = np.array(y_pred)
    if isinstance(y_true, np.ndarray) and y_true.ndim > 1:
        y_true = y_true.ravel()
    if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1:
        y_pred = y_pred.ravel()
    # Check for shape mismatch
    if y_true.shape != y_pred.shape:
        logger.error("y_true and y_pred must have the same shape.")
        raise ValueError("y_true and y_pred must have the same shape.")
    if len(y_true) == 0:
        logger.error("y_true and y_pred must not be empty.")
        raise ValueError("y_true and y_pred must not be empty.")
    if len(y_pred) == 0:
        logger.error("y_true and y_pred must not be empty.")
        raise ValueError("y_true and y_pred must not be empty.")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        logger.error("y_true and y_pred must not contain NaN values.")
        raise ValueError("y_true and y_pred must not contain NaN values.")
    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        logger.error("y_true and y_pred must not contain Inf values.")
        raise ValueError("y_true and y_pred must not contain Inf values.")
    
    # Convert to numpy arrays and flatten
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Calculate the ratio R = Z / Z0
    R = y_true / y_pred

    # Create an instance of GnosticsCharacteristics
    logger.info("Creating GnosticsCharacteristics instance.")
    gnostics = GnosticsCharacteristics(R=R)

    # Calculate q and q1
    logger.info("Calculating q and q1.")
    q, q1 = gnostics._get_q_q1()

    # Calculate fi, fj, hi, hj based on the case
    if case == 'i':
        hc = gnostics._hi(q, q1)
    
    elif case == 'j':
        hc = gnostics._hj(q, q1)
    
    else:
        logger.error("Invalid case. Use 'i' or 'j'.")
        raise ValueError("Invalid case. Use 'i' or 'j'.")

    hcsr = np.sum(hc**2)

    # normalize the result
    hcsr = hcsr / len(y_true)
    logger.info(f"Gnostic irrelevance Hc calculation completed")
    return hcsr