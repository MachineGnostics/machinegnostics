import numpy as np
import pandas as pd
from machinegnostics.magcal.util.logging import get_logger
import logging

def confusion_matrix(y_true:np.ndarray | pd.Series,
                     y_pred:np.ndarray | pd.Series, 
                     labels=None, verbose=False) -> np.ndarray:
    """
    Computes the confusion matrix to evaluate the accuracy of a classification.

    By definition, entry (i, j) in the confusion matrix is the number of observations
    actually in class i but predicted to be in class j.

    Parameters
    ----------
    y_true : array-like or pandas Series/DataFrame column of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like or pandas Series/DataFrame column of shape (n_samples,)
        Estimated targets as returned by a classifier.

    labels : array-like, default=None
        List of labels to index the matrix. This may be used to reorder or select a subset of labels.
        If None, labels that appear at least once in y_true or y_pred are used in sorted order.
    verbose : bool, optional
        If True, enables detailed logging for debugging purposes. Default is False.

    Returns
    -------
    cm : ndarray of shape (n_classes, n_classes)
        Confusion matrix whose i-th row and j-th column entry indicates the number of samples with
        true label being i-th class and predicted label being j-th class.

    Examples
    --------
    >>> y_true = [2, 0, 2, 2, 0, 1]
    >>> y_pred = [0, 0, 2, 2, 0, 2]
    >>> confusion_matrix(y_true, y_pred)
    array([[2, 0, 0],
           [0, 0, 1],
           [1, 0, 2]])
    """
    logger = get_logger('confusion_matrix', level=logging.WARNING if not verbose else logging.INFO)
    logger.info("Calculating Confusion Matrix...")
    # Convert pandas Series to numpy array
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    # Convert to numpy arrays and flatten
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if y_true.shape != y_pred.shape:
        logger.error("Shape of y_true and y_pred must be the same.")
        raise ValueError("Shape of y_true and y_pred must be the same.")
    if y_true.size == 0:
        logger.error("y_true and y_pred must not be empty.")
        raise ValueError("y_true and y_pred must not be empty.")
    # Ensure 1D arrays
    if y_true.ndim != 1 or y_pred.ndim != 1:
        logger.error("y_true and y_pred must be 1D arrays.")
        raise ValueError("y_true and y_pred must be 1D arrays.")
    # inf and nan check
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        logger.error("y_true and y_pred must not contain NaN values.")
        raise ValueError("y_true and y_pred must not contain NaN values.")
    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        logger.error("y_true and y_pred must not contain Inf values.")
        raise ValueError("y_true and y_pred must not contain Inf values.")

    # Determine labels
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.asarray(labels)
    n_labels = len(labels)
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    # Initialize confusion matrix
    cm = np.zeros((n_labels, n_labels), dtype=int)

    # Populate confusion matrix
    for true, pred in zip(y_true, y_pred):
        if true in label_to_index and pred in label_to_index:
            i = label_to_index[true]
            j = label_to_index[pred]
            cm[i, j] += 1

    logger.info("Confusion Matrix calculation completed.")
    return cm