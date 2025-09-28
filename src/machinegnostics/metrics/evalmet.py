'''
ManGo - Machine Gnostics Library
Copyright (C) 2025  ManGo Team

Author: Nirmal Parmar
'''
from machinegnostics.magcal.util.logging import get_logger
import logging
import numpy as np
from machinegnostics.magcal.criteria_eval import CriteriaEvaluator

def evalMet(y: np.ndarray, y_fit: np.ndarray, w: np.ndarray = None, verbose: bool = False) -> float:
    """
    Compute the Evaluation Metric (EvalMet) for evaluating the fit between observed data and model predictions.

    The EvalMet is a composite metric that combines Robust R-squared (RobR2), Geometric Mean of Model Fit Error (GMMFE),
    and Divergence Information (DivI) to provide a comprehensive assessment of model performance.

    Parameters
    ----------
    y : np.ndarray
        The observed data (ground truth). Must be a 1D array of numerical values.
    y_fit : np.ndarray
        The fitted data (model predictions). Must be a 1D array of the same shape as `y`.
    w : np.ndarray, optional
        Weights for the data points. If not provided, an array of ones is used.
    verbose : bool, optional
        If True, enables detailed logging for debugging purposes. Default is False.

    Returns
    -------
    float
        The computed Evaluation Metric (EvalMet) value. 

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
    - The EvalMet is calculated as:
      EvalMet = RobR2 / (GMMFE . DivI)
      where:
        - RobR2 = Robust R-squared value
        - GMMFE = Geometric Mean of Model Fit Error
        - DivI = Divergence Information

    References
    ----------
    - Kovanic P., Humber M.B (2015) The Economics of Information - Mathematical Gnostics for Data Analysis, Chapter 19.3.4

    Example
    -------
    >>> from mango.metrics.evalmet import evalMet
    >>> import numpy as np
    >>> y = np.array([
    ...     1.0, 2.0, 3.0, 4.0
    ... ])
    >>> y_fit = np.array([
    ...     1.1, 1.9, 3.2, 3.8
    ... ])
    >>> evalMet(y, y_fit, weights)
    """
    logger = get_logger('EvalMet', level=logging.WARNING if not verbose else logging.INFO)
    logger.info("Starting EvalMet calculation.")
    # Ensure y and y_fit are 1D arrays
    if y.ndim != 1 or y_fit.ndim != 1:
        logger.error("Both y and y_fit must be 1D arrays.")
        raise ValueError("Both y and y_fit must be 1D arrays.")
    
    # Ensure y and y_fit have the same shape
    if y.shape != y_fit.shape:
        logger.error("y and y_fit must have the same shape.")
        raise ValueError("y and y_fit must have the same shape.")
    
    # empty check
    if y.size == 0 or y_fit.size == 0:
        logger.error("y and y_fit must not be empty.")
        raise ValueError("y and y_fit must not be empty.")
    if np.any(np.isnan(y)) or np.any(np.isnan(y_fit)):
        logger.error("y and y_fit must not contain NaN values.")
        raise ValueError("y and y_fit must not contain NaN values.")
    if np.any(np.isinf(y)) or np.any(np.isinf(y_fit)):
        logger.error("y and y_fit must not contain Inf values.")
        raise ValueError("y and y_fit must not contain Inf values.")
    
    # If weights are not provided, use an array of ones
    if w is None:
        w = np.ones_like(y)
    
    # Ensure weights have the same shape as y
    if w.shape != y.shape:
        logger.error("Weights must have the same shape as y.")
        raise ValueError("Weights must have the same shape as y.")
    
    # Convert to numpy arrays and flatten
    y = np.asarray(y).flatten()
    y_fit = np.asarray(y_fit).flatten()
    
    # Compute the Evaluation Metric (EvalMet)
    evaluator = CriteriaEvaluator(y, y_fit, w, verbose=verbose)
    evalmet = evaluator._evalmet()
    logger.info(f"EvalMet calculation completed.")
    return evalmet
