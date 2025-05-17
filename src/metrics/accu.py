import numpy as np

def accuracy_score(y_true, y_pred):
    """
    Computes the classification accuracy.

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.

    Returns
    -------
    float
        Accuracy (0 to 1).

    Raises
    ------
    ValueError
        If y_true and y_pred have different lengths or are empty.
    TypeError
        If y_true or y_pred are not array-like.
    """
    # Validate input types
    if not isinstance(y_true, (list, tuple, np.ndarray)):
        raise TypeError("y_true must be array-like (list, tuple, or numpy array).")
    if not isinstance(y_pred, (list, tuple, np.ndarray)):
        raise TypeError("y_pred must be array-like (list, tuple, or numpy array).")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Check for matching shapes
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true shape {y_true.shape} != y_pred shape {y_pred.shape}")

    # Check for empty arrays
    if y_true.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")

    return float(np.mean(y_true == y_pred))
