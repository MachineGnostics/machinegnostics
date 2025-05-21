import numpy as np

def mean_absolute_error(y_true, y_pred):
    """
    Computes the mean absolute error (MAE).

    Parameters
    ----------
    y_true : array-like
        True values (targets).
    y_pred : array-like
        Predicted values.

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
    # Validate input types
    if not isinstance(y_true, (list, tuple, np.ndarray)):
        raise TypeError("y_true must be array-like (list, tuple, or numpy array).")
    if not isinstance(y_pred, (list, tuple, np.ndarray)):
        raise TypeError("y_pred must be array-like (list, tuple, or numpy array).")

    # Convert to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Check for shape mismatch
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true shape {y_true.shape} != y_pred shape {y_pred.shape}")

    # Check for empty arrays
    if y_true.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")

    # Compute MAE
    return float(np.mean(np.abs(y_true - y_pred)))
