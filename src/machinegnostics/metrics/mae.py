import numpy as np
from machinegnostics.magcal.util.logging import get_logger
import logging
from machinegnostics.metrics.mean import mean

def mean_absolute_error(y_true: np.ndarray, y_pred:np.ndarray, verbose: bool = False) -> float:
    """
    Computes the mean absolute error (MAE).

    Parameters
    ----------
    y_true : array-like
        True values (targets).
    y_pred : array-like
        Predicted values.
    verbose : bool, optional
        If True, enables detailed logging. Default is False.

    Returns
    -------
    float
        Average absolute difference between actual and predicted values.

    Raises
    ------
    TypeError
        If y_true or y_pred are not array-like.
    ValueError
        If inputs have mismatched shapes or are empty.
    """
    logger = get_logger('MAE', level=logging.WARNING if not verbose else logging.INFO)
    logger.info("Calculating Mean Absolute Error...")

    # Validate input types
    if not isinstance(y_true, (list, tuple, np.ndarray)):
        logger.error("Invalid input type for y_true.")
        raise TypeError("y_true must be array-like (list, tuple, or numpy array).")
    if not isinstance(y_pred, (list, tuple, np.ndarray)):
        logger.error("Invalid input type for y_pred.")
        raise TypeError("y_pred must be array-like (list, tuple, or numpy array).")
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
        logger.error(f"Shape mismatch: y_true shape {y_true.shape} != y_pred shape {y_pred.shape}")
        raise ValueError(f"Shape mismatch: y_true shape {y_true.shape} != y_pred shape {y_pred.shape}")
    # Check for empty arrays
    if y_true.size == 0:
        logger.error("y_true and y_pred must not be empty.")
        raise ValueError("y_true and y_pred must not be empty.")
    if y_pred.size == 0:
        logger.error("y_true and y_pred must not be empty.")
        raise ValueError("y_true and y_pred must not be empty.")
    if y_true.size == 0:
        logger.error("y_true and y_pred must not be empty.")
        raise ValueError("y_true and y_pred must not be empty.")

    # Convert to numpy arrays and flatten
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Compute MAE
    mae = float(mean(np.abs(y_true - y_pred)))

    logger.info(f"Mean Absolute Error (MAE) calculated.")
    return mae
