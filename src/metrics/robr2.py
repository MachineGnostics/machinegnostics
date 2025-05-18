import numpy as np
from src.magcal.criteria_eval import CriteriaEvaluator

def robr2(y: np.ndarray, y_fit: np.ndarray, w: np.ndarray = None) -> float:
    """
    Compute the Robust R-squared (RobR2) value for evaluating the goodness of fit between observed data and model predictions.

    The Robust R-squared (RobR2) is a statistical metric that measures the proportion of variance in the observed data 
    explained by the fitted data, with robustness to outliers. This function leverages the `CriteriaEvaluator` class 
    to compute the RobR2 value.

    Parameters:
        y (np.ndarray): The observed data (ground truth). Must be a 1D array.
        y_fit (np.ndarray): The fitted data (model predictions). Must be a 1D array of the same shape as `y`.
        w (np.ndarray, optional): Weights for the data points. Must be a 1D array of the same shape as `y` if provided. Defaults to `None`, in which case equal weights are assumed.

    Returns:
        float: The computed Robust R-squared (RobR2) value.

    Raises:
        ValueError: If `y` and `y_fit` do not have the same shape.
        ValueError: If `w` is provided and does not have the same shape as `y`.
        ValueError: If `y` or `y_fit` are not 1D arrays.

    Example:
        >>> import numpy as np
        >>> from src.metrics.robr2 import robr2
        >>> y = np.array([1.0, 2.0, 3.0, 4.0])
        >>> y_fit = np.array([1.1, 1.9, 3.2, 3.8])
        >>> w = np.array([1.0, 1.0, 1.0, 1.0])
        >>> result = robr2(y, y_fit, w)
        >>> print(result)
        0.98  # Example output (actual value depends on the data)

    Notes:
        - This function assumes that `y` and `y_fit` are non-negative and of the same shape.
        - The weights `w` are optional. If not provided, equal weights are assumed.
        - The function internally uses the `_robr2` method of the `CriteriaEvaluator` class to compute the metric.
    """
    # Check if y and y_fit are of the same shape
    if y.shape != y_fit.shape:
        raise ValueError("y and y_fit must have the same shape")
    
    # Check with w shape
    if w is not None and y.shape != w.shape:
        raise ValueError("y and w must have the same shape")
    
    # 1D array check
    if y.ndim != 1 or y_fit.ndim != 1:
        raise ValueError("y and y_fit must be 1D arrays")
    
    # CE
    ce = CriteriaEvaluator(y=y, y_fit=y_fit, w=w)

    # Compute the robust R-squared
    robr2_value = ce._robr2()
    return robr2_value