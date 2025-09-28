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

    Examples
    --------
    Example 1: Basic usage with simple arrays
    >>> import numpy as np
    >>> from machinegnostics.metrics import root_mean_squared_error
    >>> y_true = np.array([3, -0.5, 2, 7])
    >>> y_pred = np.array([2.5, 0.0, 2, 8])
    >>> rmse = root_mean_squared_error(y_true, y_pred, verbose=True)
    >>> print(f"RMSE: {rmse}")

    Raises
    ------
    TypeError
        If y_true or y_pred are not array-like.
    ValueError
        If inputs have mismatched shapes or are empty.
    """
    logger = get_logger('RMSE', level=logging.WARNING if not verbose else logging.INFO)
    logger.info("Calculating Root Mean Squared Error...")
    # Convert to numpy arrays and flatten
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Ensure 1D arrays (one column)
    if y_true.ndim != 1:
        logger.error("y_true must be a 1D array (single column).")
        raise ValueError("y_true must be a 1D array (single column).")
    if y_pred.ndim != 1:
        logger.error("y_pred must be a 1D array (single column).")
        raise ValueError("y_pred must be a 1D array (single column).")

    # Validate shapes
    if len(y_true) != len(y_pred):
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

    # Compute RMSE
    rmse = float(np.sqrt(mean((y_true - y_pred) ** 2)))
    logger.info(f"Gnostic RMSE calculated.")
    return rmse
