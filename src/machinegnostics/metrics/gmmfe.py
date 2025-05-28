'''
ManGo - Machine Gnostics Library
Copyright (C) 2025  ManGo Team

This work is licensed under the terms of the GNU General Public License version 3.0.
For more details, see <https://www.gnu.org/licenses/gpl-3.0.html>.
'''

import numpy as np
from machinegnostics.magcal.criteria_eval import CriteriaEvaluator

def gmmfe(y: np.ndarray, y_fit: np.ndarray) -> float:
    """
    Compute the Geometric Mean of Model Fit Error (GMMFE) for evaluating the fit between observed data and model predictions.

    The GMMFE is a statistical metric that quantifies the average relative error between the observed and fitted values 
    on a logarithmic scale. It is particularly useful for datasets with a wide range of values or when the data is 
    multiplicative in nature.

    Parameters
    ----------
    y : np.ndarray
        The observed data (ground truth). Must be a 1D array of numerical values.
    y_fit : np.ndarray
        The fitted data (model predictions). Must be a 1D array of the same shape as `y`.

    Returns
    -------
    float
        The computed Geometric Mean of Model Fit Error (GMMFE) value. 

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
    - The GMMFE is calculated using the formula:
      GMMFE = exp(Σ(w_i * log(e_i)) / Σ(w_i))
      where:
        - e_i = |y_i - y_fit_i| / |y_i| (relative error)
        - w_i = weights for each data point
      This formula computes the weighted geometric mean of the relative errors.

    References
    ----------
    - Kovanic P., Humber M.B (2015) The Economics of Information - Mathematical Gnostics for Data Analysis, Chapter 19.3.4

    Example
    -------
    >>> import numpy as np
    >>> from src.metrics.gmmfe import gmmfe
    >>> y = np.array([
    ...     1.0, 2.0, 3.0, 4.0
    ... ])
    >>> y_fit = np.array([
    ...     1.1, 1.9, 3.2, 3.8
    ... ])
    >>> gmmfe(y, y_fit)
    0.06666666666666667
    """
    # Ensure y and y_fit are 1D arrays
    if y.ndim != 1 or y_fit.ndim != 1:
        raise ValueError("y and y_fit must be 1D arrays.")
    
    # Ensure y and y_fit have the same shape
    if y.shape != y_fit.shape:
        raise ValueError("y and y_fit must have the same shape.")
    
    # Convert to numpy arrays and flatten
    y = np.asarray(y).flatten()
    y_fit = np.asarray(y_fit).flatten()
            
    # generate the GMMFE value
    ce = CriteriaEvaluator(y, y_fit)
    gmmfe_value = ce._gmmfe()    
    return gmmfe_value