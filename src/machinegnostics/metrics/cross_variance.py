'''
Gnostic Cross-Variance

Author: Nirmal Parmar
Machine Gnostics
'''

import numpy as np
from machinegnostics.magcal import EGDF, QGDF, DataHomogeneity
import logging
from machinegnostics.magcal.util.logging import get_logger

def cross_covariance(data_1: np.ndarray, data_2: np.ndarray, case: str = 'i', verbose: bool = False) -> float:
    """
    Calculate the Gnostic cross-covariance between two data samples.

    The Gnostic cross-covariance metric is based on the principles of gnostic theory, which
    provides robust estimates of data relationships. This metric leverages the concepts
    of estimating irrelevances and quantifying irrelevances, which are robust measures
    of data uncertainty. These irrelevances are aggregated differently:

    - Quantifying irrelevances are aggregated additively as hyperbolic sines.
    - Estimating irrelevances are aggregated as trigonometric sines.

    Both types of irrelevances converge to linear errors of observed data in cases of
    weak uncertainty. The cross-covariance is computed as the mean product of irrelevances
    between two data samples.

    Parameters:
    ----------
    data_1 : np.ndarray
        The first data sample. Must be a 1D numpy array without NaN or Inf values.
    data_2 : np.ndarray
        The second data sample. Must be a 1D numpy array without NaN or Inf values.
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
    >>> data_1 = np.array([1, 2, 3, 4, 5])
    >>> data_2 = np.array([5, 4, 3, 2, 1])
    >>> covar = cross_covariance(data_1, data_2, case='i', verbose=False)
    >>> print(f"Cross-Covariance (case='i'): {covar}")

    Raises:
    ------
    ValueError
        If the input arrays are not of the same length, are empty, contain NaN/Inf values,
        or are not 1D numpy arrays. Also raised if `case` is not 'i' or 'j'.

    Notes:
    -----
    - This metric is robust to data uncertainty and provides meaningful estimates even
      in the presence of noise or outliers.
    - Ensure that the input data is preprocessed and cleaned for optimal results.
    - In cases where data homogeneity is not met, a warning is raised, and the scale
      parameter is adjusted to improve results.
    """
    logger = get_logger('cross_covariance', level=logging.WARNING if not verbose else logging.INFO)
    logger.info("Starting cross-covariance computation.")
    # Validate inputs
    if len(data_1) != len(data_2):
        logger.error("Input arrays must have the same length.")
        raise ValueError("Input arrays must have the same length.")
    if len(data_1) == 0 or len(data_2) == 0:
        logger.error("Input arrays must not be empty.")
        raise ValueError("Input arrays must not be empty.")
    if not isinstance(data_1, np.ndarray) or not isinstance(data_2, np.ndarray):
        logger.error("Inputs must be numpy arrays.")
        raise ValueError("Inputs must be numpy arrays.")
    # flatten the arrays if they are not 1D
    data_1 = data_1.flatten()
    data_2 = data_2.flatten()
    if data_1.ndim != 1 or data_2.ndim != 1:
        logger.error("Input arrays must be 1D.")
        raise ValueError("Input arrays must be 1D.")
    # avoid inf and nan in data
    if np.any(np.isnan(data_1)) or np.any(np.isnan(data_2)):
        logger.error("Input arrays must not contain NaN values.")
        raise ValueError("Input arrays must not contain NaN values.")
    if np.any(np.isinf(data_1)) or np.any(np.isinf(data_2)):
        logger.error("Input arrays must not contain Inf values.")
        raise ValueError("Input arrays must not contain Inf values.")
    if case not in ['i', 'j']:
        logger.error("Case must be 'i' for estimation geometry or 'j' for quantifying geometry.")
        raise ValueError("Case must be 'i' for estimation geometry or 'j' for quantifying geometry.")

    # default arg
    FLUSH = False
    VERBOSE = False
    
    if case == 'i':
        logger.info("Using Estimation Global Distribution Function (EGDF) for correlation computation.")
        # EGDF 
        egdf_data_1 = EGDF(flush=FLUSH, verbose=VERBOSE)
        egdf_data_1.fit(data_1)

        egdf_data_2 = EGDF(flush=FLUSH, verbose=VERBOSE)
        egdf_data_2.fit(data_2)

        # Data Homogeneity
        logger.info("Performing data homogeneity check.")
        dh_data_1 = DataHomogeneity(gdf=egdf_data_1, verbose=VERBOSE, flush=FLUSH)
        is_homo_data_1 = dh_data_1.fit()

        dh_data_2 = DataHomogeneity(gdf=egdf_data_2, verbose=VERBOSE, flush=FLUSH)
        is_homo_data_2 = dh_data_2.fit()

        # check
        if not is_homo_data_1:
            logger.warning("Data 1 is not homogeneous. Switching to S=1 for better results.")
            logger.info("Fitting EGDF with S=1.")
            egdf_data_1 = EGDF(flush=FLUSH, verbose=VERBOSE, S=1)
            egdf_data_1.fit(data_1)

        if not is_homo_data_2:
            logger.warning("Data 2 is not homogeneous. Switching to S=1 for better results.")
            logger.info("Fitting EGDF with S=1.")
            egdf_data_2 = EGDF(flush=FLUSH, verbose=VERBOSE, S=1)
            egdf_data_2.fit(data_2)

        # get irrelevance of the data sample
        hc_data_1 = np.mean(egdf_data_1.hi, axis=0)
        hc_data_2 = np.mean(egdf_data_2.hi, axis=0)

    if case == 'j':
        # EGDF 
        logger.info("Using Estimation Global Distribution Function (EGDF) for correlation computation.")
        egdf_data_1 = EGDF(flush=FLUSH, verbose=VERBOSE)
        egdf_data_1.fit(data_1)

        egdf_data_2 = EGDF(flush=FLUSH, verbose=VERBOSE)
        egdf_data_2.fit(data_2)

        # data homogeneity
        logger.info("Checking data homogeneity.")
        dh_data_1 = DataHomogeneity(gdf=egdf_data_1, verbose=VERBOSE, flush=FLUSH)
        is_homo_data_1 = dh_data_1.fit()

        dh_data_2 = DataHomogeneity(gdf=egdf_data_2, verbose=VERBOSE, flush=FLUSH)
        is_homo_data_2 = dh_data_2.fit()

        # homogeneity check
        if not is_homo_data_1:
            logger.warning("Data 1 is not homogeneous. Switching to S=1 for better results.")
        if not is_homo_data_2:
            logger.warning("Data 2 is not homogeneous. Switching to S=1 for better results.")

        # QGDF
        logger.info("Using Quantification Global Distribution Function (QGDF) for correlation computation.")
        qgdf_data_1 = QGDF(flush=FLUSH, verbose=VERBOSE, S=1)
        qgdf_data_1.fit(data_1)

        qgdf_data_2 = QGDF(flush=FLUSH, verbose=VERBOSE)
        qgdf_data_2.fit(data_2)

        # get irrelevance of the data sample
        hc_data_1 = np.mean(qgdf_data_1.hj, axis=0)
        hc_data_2 = np.mean(qgdf_data_2.hj, axis=0)

        # stop overflow by limiting big value in hc up to 1e12
        hc_data_1 = np.clip(hc_data_1, 1, 1e12)
        hc_data_2 = np.clip(hc_data_2, 1, 1e12)

    cross_covar = np.mean(hc_data_1 * hc_data_2)
    logger.info(f"Cross-covariance calculated successfully.")
    return cross_covar