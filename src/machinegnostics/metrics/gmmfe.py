'''
ManGo - Machine Gnostics Library
Copyright (C) 2025  ManGo Team

Author: Nirmal Parmar
'''

import numpy as np
from machinegnostics.magcal.criteria_eval import CriteriaEvaluator
from machinegnostics.magcal.util.logging import get_logger
import logging

def gmmfe(y: np.ndarray, y_fit: np.ndarray, verbose: bool = False) -> float:
    """
    Compute the Geometric Mean of Model Fit Error (GMMFE) for evaluating the fit between observed data and model predictions.

    The GMMFE is a statistical metric that quantifies the average relative error between the observed and fitted values 
    on a logarithmic scale. It is particularly useful for datasets with a wide range of values or when the data is 
    multiplicative in nature.

    Parameters
    ----------
    y : np.ndarray
        The observed data (ground truth). Must be a 1D array of numerical values.
    y_fit : np.ndarray
        The fitted data (model predictions). Must be a 1D array of the same shape as `y`.
    verbose : bool, optional
        If True, enables detailed logging of the computation process. Default is False.

    Returns
    -------
    float
        The computed Geometric Mean of Model Fit Error (GMMFE) value. 

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
    - The GMMFE is calculated using the formula:
      GMMFE = exp(Σ(w_i * log(e_i)) / Σ(w_i))
      where:
        - e_i = |y_i - y_fit_i| / |y_i| (relative error)
        - w_i = weights for each data point
      This formula computes the weighted geometric mean of the relative errors.

    References
    ----------
    - Kovanic P., Humber M.B (2015) The Economics of Information - Mathematical Gnostics for Data Analysis, Chapter 19.3.4

    Example
    -------
    >>> import numpy as np
    >>> from src.metrics.gmmfe import gmmfe
    >>> y = np.array([
    ...     1.0, 2.0, 3.0, 4.0
    ... ])
    >>> y_fit = np.array([
    ...     1.1, 1.9, 3.2, 3.8
    ... ])
    >>> gmmfe(y, y_fit)
    0.06666666666666667
    """
    logger = get_logger('gmmfe', level=logging.WARNING if not verbose else logging.INFO)
    logger.info("Calculating GMMFE...")

    # validate input types
    if not isinstance(y, (np.ndarray,)):
        logger.error("Invalid input type for y.")
        raise ValueError("y must be a numpy array.")
    if not isinstance(y_fit, (np.ndarray,)):
        logger.error("Invalid input type for y_fit.")
        raise ValueError("y_fit must be a numpy array.")
    if y.ndim != 1 and y_fit.ndim != 1:
        logger.error("Invalid input dimensions.")
        raise ValueError("y and y_fit must be 1D arrays.")
    if y.shape != y_fit.shape:
        logger.error("Shape mismatch.")
        raise ValueError("y and y_fit must have the same shape.")
    if y.size == 0:
        logger.error("Empty array.")
        raise ValueError("y and y_fit must not be empty.")
    if y.shape != y_fit.shape:
        logger.error("Shape mismatch.")
        raise ValueError("y and y_fit must have the same shape.")
    # inf and nan check
    if np.any(np.isnan(y)) or np.any(np.isnan(y_fit)):
        logger.error("Input contains NaN values.")
        raise ValueError("y and y_fit must not contain NaN values.")
    if np.any(np.isinf(y)) or np.any(np.isinf(y_fit)):
        logger.error("Input contains Inf values.")
        raise ValueError("y and y_fit must not contain Inf values.")
    
    # Convert to numpy arrays and flatten
    y = np.asarray(y).flatten()
    y_fit = np.asarray(y_fit).flatten()
            
    # generate the GMMFE value
    ce = CriteriaEvaluator(y, y_fit, verbose=verbose)
    gmmfe_value = ce._gmmfe()    
    logger.info("GMMFE calculation completed.")
    return gmmfe_value