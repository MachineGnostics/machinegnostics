'''
ManGo - Machine Gnostics Library
Copyright (C) 2025  ManGo Team

Author: Nirmal Parmar
'''
from machinegnostics.magcal.util.logging import get_logger
import logging
import numpy as np
from machinegnostics.magcal.criteria_eval import CriteriaEvaluator

def divI(y: np.ndarray, y_fit: np.ndarray,  verbose: bool = False) -> float:
    """
    Compute the Divergence Information (DivI) for evaluating the fit between observed data and model predictions.

    The DivI is a statistical metric that measures the divergence between the distributions of the observed and fitted values
    using gnostic characteristics. It is particularly useful for assessing the quality of model fits in various applications.

    Parameters
    ----------
    y : np.ndarray
        The observed data (ground truth). Must be a 1D array of numerical values.
    y_fit : np.ndarray
        The fitted data (model predictions). Must be a 1D array of the same shape as `y`.
    verbose : bool, optional
        If True, enables detailed logging for debugging purposes. Default is False.

    Returns
    -------
    float
        The computed Divergence Information (DivI) value. 

    Raises
    ------
    ValueError
        If `y` and `y_fit` do not have the same shape.
    ValueError
        If `w` is provided and does not have the same shape as `y`.
    ValueError
        If `y` or `y_fit` are not 1D arrays.

    Notes
    -----
    - The DivI is calculated using gnostic characteristics, which provide a robust way to measure divergence between distributions.
    
    References
    ----------
    - Kovanic P., Humber M.B (2015) The Economics of Information - Mathematical Gnostics for Data Analysis, Chapter 19.3.4

    Example
    -------
    >>> import numpy as np
    >>> from src.metrics.divi import divI
    >>> y = np.array([
    ...     1.0, 2.0, 3.0, 4.0
    ... ])
    >>> y_fit = np.array([
    ...     1.1, 1.9, 3.2, 3.8
    ... ])
    >>> divI(y, y_fit)
    """
    logger = get_logger('DivI', level=logging.WARNING if not verbose else logging.INFO)
    logger.info("Starting DivI calculation.")
    # Ensure y and y_fit are 1D arrays
    if y.ndim != 1 or y_fit.ndim != 1:
        logger.error("Both y and y_fit must be 1D arrays.")
        raise ValueError("Both y and y_fit must be 1D arrays.")
    
    # Ensure y and y_fit have the same shape
    if y.shape != y_fit.shape:
        logger.error("y and y_fit must have the same shape.")
        raise ValueError("y and y_fit must have the same shape.")
    
    # Convert to numpy arrays and flatten
    y = np.asarray(y).flatten()
    y_fit = np.asarray(y_fit).flatten()
    
    # Compute the Divergence Information (DivI)
    evaluator = CriteriaEvaluator(y, y_fit, verbose=verbose)
    divI_value = evaluator._divI()
    logger.info(f"Divergence Information (DivI) calculation completed.")
    return divI_value