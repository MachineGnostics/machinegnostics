import numpy as np
import logging
from machinegnostics.magcal.util.logging import get_logger
from machinegnostics.metrics.mean import mean

def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray, verbose: bool = False) -> float:
    """
    Computes the Gnostic Root Mean Squared Error (RMSE).

    The Gnostic RMSE metric is based on the principles of gnostic theory, which
    provides robust estimates of data relationships. This metric leverages the concepts
    of estimating irrelevances and fidelities, and quantifying irrelevances and fidelities, which are robust measures of data uncertainty. These irrelevances are aggregated differently.

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
        Square root of the average of squared errors.

    Raises
    ------
    TypeError
        If y_true or y_pred are not array-like.
    ValueError
        If inputs have mismatched shapes or are empty.
    """
    logger = get_logger('RMSE', level=logging.WARNING if not verbose else logging.INFO)
    logger.info("Calculating Root Mean Squared Error...")
    # Validate input types
    if not isinstance(y_true, (list, np.ndarray)):
        logger.error("y_true must be array-like.")
        raise TypeError("y_true must be array-like.")
    if not isinstance(y_pred, (list, np.ndarray)):
        logger.error("y_pred must be array-like.")
        raise TypeError("y_pred must be array-like.")
    # Validate input shapes
    if np.ndim(y_true) > 1:
        logger.error("y_true must be a 1D array.")
        raise ValueError("y_true must be a 1D array.")
    if np.ndim(y_pred) > 1:
        logger.error("y_pred must be a 1D array.")
        raise ValueError("y_pred must be a 1D array.")
    if np.shape(y_true) != np.shape(y_pred):
        logger.error("y_true and y_pred must have the same shape.")
        raise ValueError("y_true and y_pred must have the same shape.")
    if len(y_true) == 0:
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

    # Compute RMSE
    rmse = float(np.sqrt(mean((y_true - y_pred) ** 2)))
    logger.info(f"Gnostic RMSE calculated.")
    return rmse
