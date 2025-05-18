'''
ManGo - Machine Gnostics Library
Copyright (C) 2025  ManGo Team

This work is licensed under the terms of the GNU General Public License version 3.0.
For more details, see <https://www.gnu.org/licenses/gpl-3.0.html>.
'''

import numpy as np
from src.magcal.criteria_eval import CriteriaEvaluator

def robr2(y: np.ndarray, y_fit: np.ndarray, w: np.ndarray = None) -> float:
    """
    Compute the Robust R-squared (RobR2) value for evaluating the goodness of fit between observed data and model predictions.

    The Robust R-squared (RobR2) is a statistical metric that measures the proportion of variance in the observed data 
    explained by the fitted data, with robustness to outliers. Unlike the classical R-squared metric, RobR2 incorporates 
    weights and is less sensitive to outliers, making it suitable for datasets with noise or irregularities.

    Parameters
    ----------
    y : np.ndarray
        The observed data (ground truth). Must be a 1D array of numerical values.
    y_fit : np.ndarray
        The fitted data (model predictions). Must be a 1D array of the same shape as `y`.
    w : np.ndarray, optional
        Weights for the data points. Must be a 1D array of the same shape as `y` if provided. Defaults to `None`, in which 
        case equal weights are assumed.

    Returns
    -------
    float
        The computed Robust R-squared (RobR2) value. The value ranges from 0 to 1, where:
        - 1 indicates a perfect fit.
        - 0 indicates no explanatory power of the model.

    Raises
    ------
    ValueError
        If `y` and `y_fit` do not have the same shape.
    ValueError
        If `w` is provided and does not have the same shape as `y`.
    ValueError
        If `y` or `y_fit` are not 1D arrays.

    Notes
    -----
    - The Robust R-squared (RobR2) is calculated using the formula:
      RobR2 = 1 - (Σ(w_i * (e_i - ē)²) / Σ(w_i * (y_i - ȳ)²))
      where:
        - e_i = y_i - y_fit_i (residuals)
        - ē = weighted mean of residuals
        - ȳ = weighted mean of observed data
        - w_i = weights for each data point
    - This metric is robust to outliers due to the use of weights and is particularly useful for noisy datasets.
    - If weights are not provided, equal weights are assumed for all data points.

    References
    ----------
    - Kovanic P., Humber M.B (2015) The Economics of Information - Mathematical Gnostics for Data Analysis, Chapter 19.3.4
    - Robust R-squared (RobR2) is defined in Equation 19.7 of the reference.

    Example
    -------
    >>> import numpy as np
    >>> from mango.metrics import robr2
    >>> y = np.array([1.0, 2.0, 3.0, 4.0])
    >>> y_fit = np.array([1.1, 1.9, 3.2, 3.8])
    >>> w = np.array([1.0, 1.0, 1.0, 1.0])
    >>> result = robr2(y, y_fit, w)
    >>> print(result)
    0.98  # Example output (actual value depends on the data)

    Comparison with Classical R-squared
    -----------------------------------
    The classical R-squared metric assumes equal weights and is sensitive to outliers. RobR2, on the other hand, 
    incorporates weights and is robust to outliers, making it more reliable for datasets with irregularities or noise.
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