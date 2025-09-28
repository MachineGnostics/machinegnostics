'''
Gnostic Correlation Metric

This module provides a function to compute the Gnostic correlation between two data samples.

Author: Nirmal Parmar
Machine Gnostics
'''

import numpy as np
from machinegnostics.magcal import EGDF, QGDF, DataHomogeneity
import logging
from machinegnostics.magcal.util.logging import get_logger

def correlation(X: np.ndarray, y: np.ndarray, case: str = 'i', verbose: bool = False) -> float:
    """
    Calculate the Gnostic correlation coefficient between a feature array X and a target array y.

    Parameters:
    ----------
    X : np.ndarray
        The feature data sample. Must be a numpy array without NaN or Inf values.
        If X has more than one column, pass each column one by one to this function.
    y : np.ndarray
        The target data sample. Must be a 1D numpy array without NaN or Inf values.
    case : str, optional, default='i'
        Specifies the type of geometry to use:
        - 'i': Estimation geometry (EGDF).
        - 'j': Quantifying geometry (QGDF).
    verbose : bool, optional, default=False
        If True, enables detailed logging for debugging purposes.

    Returns:
    -------
    float
        The Gnostic correlation coefficient between the two data samples.

    Examples:
    ---------
    Example 1: Compute correlation for two simple datasets
    >>> import numpy as np
    >>> from machinegnostics.metrics import correlation
    >>> X = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([5, 4, 3, 2, 1])
    >>> corr = correlation(X, y, case='i', verbose=False)
    >>> print(f"Correlation (case='i'): {corr}")

    Example 2: For multi-column X
    >>> X = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]])
    >>> y = np.array([5, 4, 3, 2, 1])
    >>> for i in range(X.shape[1]):
    ...     corr = correlation(X[:, i], y)
    ...     print(f"Correlation for column {i}: {corr}")

    Raises:
    ------
    ValueError
        If the input arrays are not of the same length, are empty, contain NaN/Inf values,
        or are not 1D numpy arrays. Also raised if `case` is not 'i' or 'j'.

    Notes:
    -----
    - If X has more than one column, pass each column separately (e.g., X[:, i]).
    - y must be a 1D array.
    - This metric is robust to data uncertainty and provides meaningful estimates even
      in the presence of noise or outliers.
    - Ensure that the input data is preprocessed and cleaned for optimal results.
    - In cases where data homogeneity is not met, a warning is raised, and the scale
      parameter is adjusted to improve results.
    """
    logger = get_logger('correlation', level=logging.WARNING if not verbose else logging.INFO)
    logger.info("Starting correlation computation.")

    # Validate inputs
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        logger.error("Inputs must be numpy arrays.")
        raise ValueError("Inputs must be numpy arrays.")

    # Flatten X and y to 1D if possible
    X = X.flatten()
    y = y.flatten()

    if len(X) != len(y):
        logger.error("Input arrays must have the same length.")
        raise ValueError("Input arrays must have the same length.")
    if len(X) == 0 or len(y) == 0:
        logger.error("Input arrays must not be empty.")
        raise ValueError("Input arrays must not be empty.")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        logger.error("Input arrays must not contain NaN values.")
        raise ValueError("Input arrays must not contain NaN values.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        logger.error("Input arrays must not contain Inf values.")
        raise ValueError("Input arrays must not contain Inf values.")
    if case not in ['i', 'j']:
        logger.error("Case must be 'i' for estimation geometry or 'j' for quantifying geometry.")
        raise ValueError("Case must be 'i' for estimation geometry or 'j' for quantifying geometry.")

    # default arg
    FLUSH = False
    VERBOSE = False

    # ...existing code logic, replacing data_1 with X and data_2 with y...
    if case == 'i':
        logger.info("Using Estimation Global Distribution Function (EGDF) for correlation computation.")
        egdf_X = EGDF(flush=FLUSH, verbose=VERBOSE)
        egdf_X.fit(X)

        egdf_y = EGDF(flush=FLUSH, verbose=VERBOSE)
        egdf_y.fit(y)

        logger.info("Performing data homogeneity check.")
        dh_X = DataHomogeneity(gdf=egdf_X, verbose=VERBOSE, flush=FLUSH)
        is_homo_X = dh_X.fit()

        dh_y = DataHomogeneity(gdf=egdf_y, verbose=VERBOSE, flush=FLUSH)
        is_homo_y = dh_y.fit()

        if not is_homo_X:
            logger.warning("X is not homogeneous. Switching to S=1 for better results.")
            logger.info("Fitting EGDF with S=1.")
            egdf_X = EGDF(flush=FLUSH, verbose=VERBOSE, S=1)
            egdf_X.fit(X)

        if not is_homo_y:
            logger.warning("y is not homogeneous. Switching to S=1 for better results.")
            logger.info("Fitting EGDF with S=1.")
            egdf_y = EGDF(flush=FLUSH, verbose=VERBOSE, S=1)
            egdf_y.fit(y)

        hc_X = np.mean(egdf_X.hi, axis=0)
        hc_y = np.mean(egdf_y.hi, axis=0)

    if case == 'j':
        logger.info("Using Estimation Global Distribution Function (EGDF) for correlation computation.")
        egdf_X = EGDF(flush=FLUSH, verbose=VERBOSE)
        egdf_X.fit(X)

        egdf_y = EGDF(flush=FLUSH, verbose=VERBOSE)
        egdf_y.fit(y)

        logger.info("Checking data homogeneity.")
        dh_X = DataHomogeneity(gdf=egdf_X, verbose=VERBOSE, flush=FLUSH)
        is_homo_X = dh_X.fit()

        dh_y = DataHomogeneity(gdf=egdf_y, verbose=VERBOSE, flush=FLUSH)
        is_homo_y = dh_y.fit()

        if not is_homo_X:
            logger.warning("X is not homogeneous. Switching to S=1 for better results.")
        if not is_homo_y:
            logger.warning("y is not homogeneous. Switching to S=1 for better results.")

        logger.info("Using Quantification Global Distribution Function (QGDF) for correlation computation.")
        qgdf_X = QGDF(flush=FLUSH, verbose=VERBOSE, S=1)
        qgdf_X.fit(X)

        qgdf_y = QGDF(flush=FLUSH, verbose=VERBOSE)
        qgdf_y.fit(y)

        hc_X = np.mean(qgdf_X.hj, axis=0)
        hc_y = np.mean(qgdf_y.hj, axis=0)

        hc_X = np.clip(hc_X, 1, 1e12)
        hc_y = np.clip(hc_y, 1, 1e12)

    def compute_correlation(hc_X: np.ndarray, hc_y: np.ndarray) -> float:
        logger.info("Computing correlation.")
        numerator = np.sum(hc_X * hc_y)
        denominator = (np.sqrt(np.sum(hc_X**2)) * np.sqrt(np.sum(hc_y**2))) 
        corr = numerator / denominator
        if denominator == 0:
            return np.nan
        return corr

    corr = compute_correlation(hc_X, hc_y)
    logger.info("Correlation computed successfully.")
    return corr