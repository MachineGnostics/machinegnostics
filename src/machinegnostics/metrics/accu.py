import numpy as np

def accuracy_score(y_true:np.ndarray, y_pred:np.ndarray) -> float:
    """
    Computes the classification accuracy.

    The classification accuracy is the ratio of correctly predicted class labels to the total number of predictions. 
    It is a commonly used metric for evaluating the performance of classification models. The accuracy score ranges 
    from 0 to 1, where:
    - 1 indicates perfect accuracy (all predictions are correct).
    - 0 indicates no correct predictions.

    Parameters
    ----------
    y_true : array-like
        True class labels. Must be a 1D array-like object (e.g., list, tuple, or numpy array) containing the ground truth labels.
    y_pred : array-like
        Predicted class labels. Must be a 1D array-like object (e.g., list, tuple, or numpy array) containing the predicted labels.

    Returns
    -------
    float
        The classification accuracy as a float value between 0 and 1.

    Raises
    ------
    ValueError
        - If `y_true` and `y_pred` have different lengths.
        - If `y_true` or `y_pred` are empty.
    TypeError
        - If `y_true` or `y_pred` are not array-like (e.g., list, tuple, or numpy array).

    Notes
    -----
    - The function converts `y_true` and `y_pred` to numpy arrays internally for efficient computation.
    - The comparison `y_true == y_pred` is performed element-wise, and the mean of the resulting boolean array is computed to determine accuracy.

    - The function does not handle multi-class classification or multi-label classification scenarios. It assumes binary classification.

    - Made for research purposes, and may not be suitable for production use without further validation and testing.

    """
    # Validate input types
    if not isinstance(y_true, (list, tuple, np.ndarray)):
        raise TypeError("y_true must be array-like (list, tuple, or numpy array).")
    if not isinstance(y_pred, (list, tuple, np.ndarray)):
        raise TypeError("y_pred must be array-like (list, tuple, or numpy array).")

    # Convert to numpy arrays and flatten
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Check for matching shapes
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true shape {y_true.shape} != y_pred shape {y_pred.shape}")

    # Check for empty arrays
    if y_true.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")

    return float(np.mean(y_true == y_pred))
