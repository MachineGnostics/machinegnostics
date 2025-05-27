'''
ManGo - Machine Gnostics Library
Copyright (C) 2025  ManGo Team

This work is licensed under the terms of the GNU General Public License version 3.0.
For more details, see <https://www.gnu.org/licenses/gpl-3.0.html>.
'''

import numpy as np
import pandas as pd
from machinegnostics.magcal import __gcorrelation


def gcorrelation(data_1: np.ndarray, data_2: np.ndarray) -> np.ndarray:
    """
    Calculate the Gnostic correlation between two data samples using robust irrelevance-based weighting.

    This function implements the robust gnostic correlation as described in Kovanic & Humber (2015).
    The method uses irrelevance functions to construct weights,
    providing a robust alternative to classical Pearson correlation. It is less sensitive to outliers,
    does not assume normality.

    Parameters
    ----------
    data_1 : np.ndarray
        First data sample (1D array).
    data_2 : np.ndarray
        Second data sample to compare with data_1 (must be same length as data_1).

    Returns
    -------
    float
        The calculated Gnostic correlation coefficient.
        For case='i': Range [-1, 1] (robust analog of Pearson correlation).
        For case='j': Range [0, ∞) (measures strength of relationship).

    Raises
    ------
    ValueError
        If input arrays have different lengths, are empty, are not numpy arrays, or if case is invalid.

    Notes
    -----
    - Gnostic correlation is robust against outliers and non-normal data.
    - Uses irrelevance functions to construct weights, following the gnostic framework.
    - The location parameter is set by the mean (can be replaced by G-median for higher robustness).
    - The geometric mean of the weights is used as the "best" weighting vector, as per the reference.
    - For 2D arrays, apply this function column-wise.

    References
    ----------
    .. [1] Kovanic P., Humber M.B (2015) The Economics of Information - Mathematical
           Gnostics for Data Analysis, Chapter 14.3.3, Equation 24.7

    Examples
    --------
    >>> import numpy as np
    >>> from machinegnostics.metrics import gcorrelation
    >>> # Example 1: Linear relationship with noise (estimation case)
    >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> y = np.array([0.9, 2.1, 2.9, 4.2, 4.8])
    >>> gcor = gcorrelation(x, y)
    >>> print(f"Estimation correlation: {gcor:.3f}")
    Estimation correlation: 0.999
 
    References
    ----------
    .. [1] Kovanic P., Humber M.B (2015) The Economics of Information - Mathematical
           Gnostics for Data Analysis, Chapter 14
    """
    # Save original column names if pandas
    x_names = None
    y_names = None
    is_pandas = False

    if isinstance(data_1, pd.DataFrame):
        x_names = data_1.columns.tolist()
        data_1 = data_1.values
        is_pandas = True
    elif isinstance(data_1, pd.Series):
        x_names = [data_1.name if data_1.name is not None else "x"]
        data_1 = data_1.values
        is_pandas = True

    if isinstance(data_2, pd.DataFrame):
        y_names = data_2.columns.tolist()
        data_2 = data_2.values
        is_pandas = True
    elif isinstance(data_2, pd.Series):
        y_names = [data_2.name if data_2.name is not None else "y"]
        data_2 = data_2.values
        is_pandas = True

    # check if inputs are numpy arrays
    if not isinstance(data_1, np.ndarray) or not isinstance(data_2, np.ndarray):
        raise ValueError("Input data must be numpy arrays or pandas DataFrame/Series.")

    # Convert 1D to 2D (variables as columns)
    if data_1.ndim == 1:
        data_1 = data_1.reshape(-1, 1)
    if data_2.ndim == 1:
        data_2 = data_2.reshape(-1, 1)

    if data_1.shape[0] != data_2.shape[0]:
        raise ValueError("Each row in data_1 and data_2 must have the same number of samples (columns).")

    n_x = data_1.shape[1]
    n_y = data_2.shape[1]
    corr_matrix = np.zeros((n_x, n_y))

    for i in range(n_x):
        for j in range(n_y):
            corr_matrix[i, j] = __gcorrelation(data_1[:, i], data_2[:, j])
    
    # If input was pandas, return DataFrame with column names
    if is_pandas:
        if x_names is None:
            x_names = [f"x{i}" for i in range(n_x)]
        if y_names is None:
            y_names = [f"y{j}" for j in range(n_y)]
        df_corr = pd.DataFrame(corr_matrix, index=x_names, columns=y_names)
        return df_corr

    return corr_matrix