import numpy as np
import pandas as pd
import logging
from machinegnostics.magcal.util.logging import get_logger
from machinegnostics.metrics.mean import mean

def accuracy_score(y_true:np.ndarray, y_pred:np.ndarray, verbose:bool=False) -> float:
    """
    Computes the accuracy classification score.

    Supports input as numpy arrays, lists, or pandas Series/DataFrame columns.

    Parameters
    ----------
    y_true : array-like or pandas Series/DataFrame column of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like or pandas Series/DataFrame column of shape (n_samples,)
        Estimated targets as returned by a classifier.
    verbose : bool, optional
        If True, enables detailed logging for debugging purposes. Default is False.

    Returns
    -------
    accuracy : float
        The accuracy score as a float in the range [0, 1].

    Examples
    --------
    >>> y_true = [0, 1, 2, 2, 0]
    >>> y_pred = [0, 0, 2, 2, 0]
    >>> accuracy_score(y_true, y_pred)
    0.8

    >>> import pandas as pd
    >>> df = pd.DataFrame({'true': [1, 0, 1], 'pred': [1, 1, 1]})
    >>> accuracy_score(df['true'], df['pred'])
    """
    logger = get_logger('accuracy_score', level=logging.WARNING if not verbose else logging.INFO)
    logger.info("Calculating Accuracy Score...")
    # Check for empty input
    if y_true is None or y_pred is None:
        logger.error("y_true and y_pred must not be None.")
        raise ValueError("y_true and y_pred must not be None.")
    # If input is a DataFrame, raise error (must select column)
    if isinstance(y_true, pd.DataFrame) or isinstance(y_pred, pd.DataFrame):
        logger.error("y_true and y_pred must be 1D array-like or pandas Series, not DataFrame. Select a column.")
        raise ValueError("y_true and y_pred must be 1D array-like or pandas Series, not DataFrame. Select a column.")

    # Convert pandas Series to numpy array
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    # Convert to numpy arrays and flatten
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if y_true.shape != y_pred.shape:
        raise ValueError("Shape of y_true and y_pred must be the same.")

    correct = np.sum(y_true == y_pred)
    total = y_true.size
    accuracy = correct / total
    logger.info("Accuracy score calculation complete.")
    return accuracy