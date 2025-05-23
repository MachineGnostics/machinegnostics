'''
ManGo - Machine Gnostics Library
Copyright (C) 2025  ManGo Team

This work is licensed under the terms of the GNU General Public License version 3.0.
For more details, see <https://www.gnu.org/licenses/gpl-3.0.html>.
'''

import numpy as np
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
    >>> from machinegnostics.magcal.gcor import gcorrelation
    >>> # Example 1: Linear relationship with noise (estimation case)
    >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> y = np.array([0.9, 2.1, 2.9, 4.2, 4.8])
    >>> gcor = gcorrelation(x, y, case='i')
    >>> print(f"Estimation correlation: {gcor:.3f}")
    Estimation correlation: 0.999
    >>> # Example 2: Inherent variability analysis (quantification case)
    >>> measurements_A = np.array([10.1, 10.3, 9.8, 10.2, 10.0])
    >>> measurements_B = np.array([5.1, 5.2, 4.9, 5.3, 5.0])
    >>> gcor = gcorrelation(measurements_A, measurements_B, case='j')
    >>> print(f"Quantification correlation: {gcor:.3f}")
    Quantification correlation: 0.999
 
    References
    ----------
    .. [1] Kovanic P., Humber M.B (2015) The Economics of Information - Mathematical
           Gnostics for Data Analysis, Chapter 14.3.3
    """
    if data_1.ndim == 1:
        data_1 = data_1[np.newaxis, :]
    if data_2.ndim == 1:
        data_2 = data_2[np.newaxis, :]
    if data_1.shape[1] != data_2.shape[1]:
        raise ValueError("Each row in data_1 and data_2 must have the same number of samples (columns).")

    n_x, n_samples = data_1.shape
    n_y = data_2.shape[0]
    corr_matrix = np.zeros((n_x, n_y))

    for i in range(n_x):
        for j in range(n_y):
            corr_matrix[i, j] = __gcorrelation(data_1[i], data_2[j])
    return corr_matrix