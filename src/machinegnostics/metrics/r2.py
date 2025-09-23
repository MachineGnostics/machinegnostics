import numpy as np
from machinegnostics.magcal.util.logging import get_logger
import logging

def r2_score(y_true:np.ndarray, y_pred:np.ndarray, verbose:bool=False) -> float:
    """
    Computes the coefficient of determination (R² score).

    Parameters
    ----------
    y_true : array-like
        True values (targets).
    y_pred : array-like
        Predicted values.
    verbose : bool, optional
        If True, enables detailed logging for debugging purposes. Default is False.

    Returns
    -------
    float
        Proportion of variance explained (1 is perfect prediction).

    Raises
    ------
    TypeError
        If inputs are not array-like.
    ValueError
        If shapes do not match or inputs are empty.
    """
    logger = get_logger('R2', level=logging.WARNING if not verbose else logging.INFO)
    logger.info("Calculating R2 Score...")

    if not isinstance(y_true, (list, tuple, np.ndarray)):
        logger.error("y_true must be array-like.")
        raise TypeError("y_true must be array-like.")
    if not isinstance(y_pred, (list, tuple, np.ndarray)):
        logger.error("y_pred must be array-like.")
        raise TypeError("y_pred must be array-like.")

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if y_true.shape != y_pred.shape:
        logger.error(f"Shape mismatch: y_true shape {y_true.shape} != y_pred shape {y_pred.shape}")
        raise ValueError(f"Shape mismatch: y_true shape {y_true.shape} != y_pred shape {y_pred.shape}")

    if y_true.size == 0:
        logger.error("y_true and y_pred must not be empty.")
        raise ValueError("y_true and y_pred must not be empty.")

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        # All values in y_true are identical; R² is not defined
        return 0.0
    logger.info("R2 Score calculated.")

    return float(1 - ss_res / ss_tot)

def adjusted_r2_score(y_true:np.ndarray, y_pred:np.ndarray, n_features:int, verbose:bool=False) -> float:
    """
    Computes the adjusted R² score.

    Parameters
    ----------
    y_true : array-like
        True values (targets).
    y_pred : array-like
        Predicted values.
    n_features : int
        Number of features (independent variables) in the model.
    verbose : bool, optional
        If True, enables detailed logging for debugging purposes. Default is False.

    Returns
    -------
    float
        Adjusted R² accounting for number of predictors.

    Raises
    ------
    ValueError
        If n_features is invalid (e.g., greater than or equal to number of samples).
    """
    logger = get_logger('adjusted_R2', level=logging.WARNING if not verbose else logging.INFO)
    logger.info("Calculating Adjusted R2 Score...")

    if not isinstance(n_features, int) or n_features < 0:
        logger.error("n_features must be a non-negative integer.")
        raise ValueError("n_features must be a non-negative integer.")

    # Convert to numpy arrays and flatten
    if not isinstance(y_true, (list, tuple, np.ndarray)):
        logger.error("y_true must be array-like.")
        raise TypeError("y_true must be array-like.")
    if not isinstance(y_pred, (list, tuple, np.ndarray)):
        logger.error("y_pred must be array-like.")
        raise TypeError("y_pred must be array-like.")
    y_pred = np.asarray(y_pred).flatten()
    y_true = np.asarray(y_true).flatten()
    if y_true.shape != y_pred.shape:
        logger.error(f"Shape mismatch: y_true shape {y_true.shape} != y_pred shape {y_pred.shape}")
        raise ValueError(f"Shape mismatch: y_true shape {y_true.shape} != y_pred shape {y_pred.shape}")
    if y_true.size == 0:
        logger.error("y_true and y_pred must not be empty.")
        raise ValueError("y_true and y_pred must not be empty.")
    
    
    n = y_true.shape[0]

    if n <= n_features + 1:
        raise ValueError(
            f"Adjusted R² is undefined for n = {n} and n_features = {n_features} "
            "(must have n > n_features + 1)."
        )

    r2 = r2_score(y_true, y_pred)
    r2_adj = float(1 - (1 - r2) * (n - 1) / (n - n_features - 1))

    logger.info("Adjusted R2 Score calculated.")
    return r2_adj
