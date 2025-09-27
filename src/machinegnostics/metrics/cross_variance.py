'''
Gnostic Cross-Variance

Author: Nirmal Parmar
Machine Gnostics
'''

import numpy as np
from machinegnostics.magcal import EGDF, QGDF, DataHomogeneity
import logging
from machinegnostics.magcal.util.logging import get_logger

def cross_covariance(X: np.ndarray, y: np.ndarray, case: str = 'i', verbose: bool = False) -> float:
    """
    Calculate the Gnostic cross-covariance between a feature array X and a target array y.

    Parameters:
    ----------
    X : np.ndarray
        The feature data sample. Must be a 1D numpy array (single feature/column).
        If X has more than one column, pass each column separately (e.g., X[:, i]).
    y : np.ndarray
        The target data sample. Must be a 1D numpy array without NaN or Inf values.
    case : str, optional, default='i'
        Specifies the type of geometry to use:
        - 'i': Estimation geometry.
        - 'j': Quantifying geometry.
    verbose : bool, optional, default=False
        If True, enables detailed logging for debugging purposes.

    Returns:
    -------
    float
        The Gnostic cross-covariance between the two data samples.

    Examples:
    ---------
    Example 1: Compute cross-covariance for two simple datasets
    >>> import numpy as np
    >>> from machinegnostics.metrics import cross_covariance
    >>> X = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([5, 4, 3, 2, 1])
    >>> covar = cross_covariance(X, y, case='i', verbose=False)
    >>> print(f"Cross-Covariance (case='i'): {covar}")

    Example 2: For multi-column X
    >>> X = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]])
    >>> y = np.array([5, 4, 3, 2, 1])
    >>> for i in range(X.shape[1]):
    ...     covar = cross_covariance(X[:, i], y)
    ...     print(f"Cross-Covariance for column {i}: {covar}")

    Raises:
    ------
    ValueError
        If the input arrays are not of the same length, are empty, contain NaN/Inf values,
        or are not 1D numpy arrays. Also raised if `case` is not 'i' or 'j'.

    Notes:
    -----
    - X must be a 1D numpy array (single column). For multi-column X, pass each column separately.
    - y must be a 1D numpy array.
    - This metric is robust to data uncertainty and provides meaningful estimates even
      in the presence of noise or outliers.
    - Ensure that the input data is preprocessed and cleaned for optimal results.
    - In cases where data homogeneity is not met, a warning is raised, and the scale
      parameter is adjusted to improve results.
    """
    logger = get_logger('cross_covariance', level=logging.WARNING if not verbose else logging.INFO)
    logger.info("Starting cross-covariance computation.")
    # Validate inputs
    if len(X) != len(y):
        logger.error("Input arrays must have the same length.")
        raise ValueError("Input arrays must have the same length.")
    if len(X) == 0 or len(y) == 0:
        logger.error("Input arrays must not be empty.")
        raise ValueError("Input arrays must not be empty.")
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        logger.error("Inputs must be numpy arrays.")
        raise ValueError("Inputs must be numpy arrays.")
    # flatten the arrays if they are not 1D
    X = X.flatten()
    y = y.flatten()
    if X.ndim != 1 or y.ndim != 1:
        logger.error("X and y must be 1D numpy arrays. For multi-column X, pass each column separately (e.g., X[:, i]).")
        raise ValueError("X and y must be 1D numpy arrays. For multi-column X, pass each column separately (e.g., X[:, i]).")
    # avoid inf and nan in data
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        logger.error("Input arrays must not contain NaN values.")
        raise ValueError("Input arrays must not contain NaN values.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        logger.error("Input arrays must not contain Inf values.")
        raise ValueError("Input arrays must not contain Inf values.")
    if case not in ['i', 'j']:
        logger.error("Case must be 'i' for estimation geometry or 'j' for quantifying geometry.")
        raise ValueError("Case must be 'i' for estimation geometry or 'j' for quantifying geometry.")

    # ...existing logic unchanged...
    FLUSH = False
    VERBOSE = False
    
    if case == 'i':
        logger.info("Using Estimation Global Distribution Function (EGDF) for correlation computation.")
        egdf_data_1 = EGDF(flush=FLUSH, verbose=VERBOSE)
        egdf_data_1.fit(X)

        egdf_data_2 = EGDF(flush=FLUSH, verbose=VERBOSE)
        egdf_data_2.fit(y)

        logger.info("Performing data homogeneity check.")
        dh_data_1 = DataHomogeneity(gdf=egdf_data_1, verbose=VERBOSE, flush=FLUSH)
        is_homo_data_1 = dh_data_1.fit()

        dh_data_2 = DataHomogeneity(gdf=egdf_data_2, verbose=VERBOSE, flush=FLUSH)
        is_homo_data_2 = dh_data_2.fit()

        if not is_homo_data_1:
            logger.warning("X is not homogeneous. Switching to S=1 for better results.")
            logger.info("Fitting EGDF with S=1.")
            egdf_data_1 = EGDF(flush=FLUSH, verbose=VERBOSE, S=1)
            egdf_data_1.fit(X)

        if not is_homo_data_2:
            logger.warning("y is not homogeneous. Switching to S=1 for better results.")
            logger.info("Fitting EGDF with S=1.")
            egdf_data_2 = EGDF(flush=FLUSH, verbose=VERBOSE, S=1)
            egdf_data_2.fit(y)

        hc_data_1 = np.mean(egdf_data_1.hi, axis=0)
        hc_data_2 = np.mean(egdf_data_2.hi, axis=0)

    if case == 'j':
        logger.info("Using Estimation Global Distribution Function (EGDF) for correlation computation.")
        egdf_data_1 = EGDF(flush=FLUSH, verbose=VERBOSE)
        egdf_data_1.fit(X)

        egdf_data_2 = EGDF(flush=FLUSH, verbose=VERBOSE)
        egdf_data_2.fit(y)

        logger.info("Checking data homogeneity.")
        dh_data_1 = DataHomogeneity(gdf=egdf_data_1, verbose=VERBOSE, flush=FLUSH)
        is_homo_data_1 = dh_data_1.fit()

        dh_data_2 = DataHomogeneity(gdf=egdf_data_2, verbose=VERBOSE, flush=FLUSH)
        is_homo_data_2 = dh_data_2.fit()

        if not is_homo_data_1:
            logger.warning("X is not homogeneous. Switching to S=1 for better results.")
        if not is_homo_data_2:
            logger.warning("y is not homogeneous. Switching to S=1 for better results.")

        logger.info("Using Quantification Global Distribution Function (QGDF) for correlation computation.")
        qgdf_data_1 = QGDF(flush=FLUSH, verbose=VERBOSE, S=1)
        qgdf_data_1.fit(X)

        qgdf_data_2 = QGDF(flush=FLUSH, verbose=VERBOSE)
        qgdf_data_2.fit(y)

        hc_data_1 = np.mean(qgdf_data_1.hj, axis=0)
        hc_data_2 = np.mean(qgdf_data_2.hj, axis=0)

        hc_data_1 = np.clip(hc_data_1, 1, 1e12)
        hc_data_2 = np.clip(hc_data_2, 1, 1e12)

    cross_covar = np.mean(hc_data_1 * hc_data_2)
    logger.info(f"Cross-covariance calculated successfully.")
    return cross_covar