import numpy as np
from machinegnostics.magcal.util.logging import get_logger
import logging
from machinegnostics.metrics.mean import mean

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray, verbose: bool = False) -> float:
    """
    Computes the Gnostic mean squared error (MSE).

    The Gnostic MSE metric is based on the principles of gnostic theory, which
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
        Average of squared differences between actual and predicted values.

    Raises
    ------
    TypeError
        If y_true or y_pred are not array-like.
    ValueError
        If inputs have mismatched shapes or are empty.
    """
    logger = get_logger('MSE', level=logging.WARNING if not verbose else logging.INFO)
    logger.info("Calculating Mean Squared Error...")

    # Validate input types
    if not isinstance(y_true, (list, tuple, np.ndarray)):
        logger.error("y_true must be array-like (list, tuple, or numpy array).")
        raise TypeError("y_true must be array-like (list, tuple, or numpy array).")
    if not isinstance(y_pred, (list, tuple, np.ndarray)):
        logger.error("y_pred must be array-like (list, tuple, or numpy array).")
        raise TypeError("y_pred must be array-like (list, tuple, or numpy array).")
    # Validate input dimensions
    if np.ndim(y_true) > 1:
        logger.error("y_true must be a 1D array.")
        raise ValueError("y_true must be a 1D array.")
    if np.ndim(y_pred) > 1:
        logger.error("y_pred must be a 1D array.")
        raise ValueError("y_pred must be a 1D array.")
    # Check for shape mismatch
    if np.shape(y_true) != np.shape(y_pred):
        logger.error("y_true and y_pred must have the same shape.")
        raise ValueError("y_true and y_pred must have the same shape.")
    # Check for empty arrays
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

    # Check for empty arrays
    if y_true.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")

    # Compute MSE
    mse = float(mean((y_true - y_pred) ** 2))
    logger.info(f"Gnostic MSE calculated.")
    return mse
