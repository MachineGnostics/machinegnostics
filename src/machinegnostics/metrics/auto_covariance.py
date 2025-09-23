'''
Gnostic auto-co-variance

Author: Nirmal Parmar
Machine Gnostics
'''

from re import VERBOSE
import numpy as np
from machinegnostics.magcal import EGDF, QGDF, DataHomogeneity
from machinegnostics.magcal.util.logging import get_logger
import logging

def auto_covariance(data: np.ndarray, lag: int = 0, case: str = 'i', verbose: bool = False) -> float:
    """
    Calculate the Gnostic auto-covariance of a data sample.

    Auto-covariance measures the relationship between a data sample and a lagged version of itself.
    This function uses the principles of Gnostic theory to compute robust estimates of auto-covariance.

    Parameters:
    ----------
    data : np.ndarray
        The data sample. Must be a 1D numpy array without NaN or Inf values.
        The input data should represent a time series or sequential data points.
    lag : int, optional, default=0
        The lag value for which the auto-covariance is computed. Must be non-negative and less than the length of the data.
        A lag of 0 computes the covariance of the data with itself.
    case : str, optional, default='i'
        Specifies the type of geometry to use for irrelevance computation:
        - 'i': Estimation Geometry Distribution Function.
        - 'j': Quantifying Geometry Distribution Function.
    verbose : bool, optional, default=False
        If True, detailed logging information will be printed during the computation.

    Returns:
    -------
    float
        The Gnostic auto-covariance coefficient for the given lag. If the computed value is less than 1e-6, it is set to 0.0.

    Raises:
    ------
    ValueError
        If the input array is invalid (e.g., not a numpy array, contains NaN/Inf values, is not 1D, or is empty).
        If the lag is negative or greater than or equal to the length of the data.
        If the case is not one of ['i', 'j'].

    Notes:
    -----
    - This function uses Gnostic theory to compute irrelevance values for the data and its lagged version.
    - Irrelevance values are clipped to avoid overflow, with a maximum value of 1e12.
    - Homogeneity checks are performed on the data and its lagged version. If the data is not homogeneous, warnings are raised.

    Warnings:
    --------
    - If the data or its lagged version is not homogeneous, a warning is printed suggesting the use of a scale parameter ( S = 1 ) for better results.

    Examples:
    ---------
    Example 1: Compute auto-covariance for a simple dataset
    >>> from machinegnostics.metrics import auto_covariance
    >>> import machinegnostics as mg # alternative import
    >>> import numpy as np
    >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> lag = 1
    >>> auto_covar = auto_covariance(data, lag=lag, case='i')
    >>> auto_covar = mg.auto_covariance(data, lag=lag, case='i') # alternative usage
    >>> print(f"Auto-covariance with lag={lag}: {auto_covar}")

    Example 2: Compute auto-covariance for a dataset with QGDF
    >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> lag = 2
    >>> auto_covar = auto_covariance(data, lag=lag, case='j')
    >>> print(f"Auto-covariance with lag={lag}: {auto_covar}")

    Example 3: Handle invalid input
    >>> data = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
    >>> lag = 1
    >>> try:
    >>>     auto_covar = auto_covariance(data, lag=lag, case='i')
    >>> except ValueError as e:
    >>>     print(f"Error: {e}")

    """
    logger = get_logger('auto_covariance', level=logging.WARNING if not verbose else logging.INFO)
    # Validate inputs
    if not isinstance(data, np.ndarray):
        logger.error("Input must be a numpy array.")
        raise ValueError("Input must be a numpy array.")
    # flatten data
    data = data.flatten()
    if data.ndim != 1:
        logger.error("Input array must be 1D.")
        raise ValueError("Input array must be 1D.")
    if len(data) == 0:
        logger.error("Input array must not be empty.")
        raise ValueError("Input array must not be empty.")
    if np.any(np.isnan(data)):
        logger.error("Input array must not contain NaN values.")
        raise ValueError("Input array must not contain NaN values.")
    if np.any(np.isinf(data)):
        logger.error("Input array must not contain Inf values.")
        raise ValueError("Input array must not contain Inf values.")
    if lag < 0 or lag >= len(data):
        logger.error("Lag must be non-negative and less than the length of the data.")
        raise ValueError("Lag must be non-negative and less than the length of the data.")
    if case not in ['i', 'j']:
        logger.error("Case must be 'i' for estimation geometry or 'j' for quantifying geometry.")
        raise ValueError("Case must be 'i' for estimation geometry or 'j' for quantifying geometry.")

    # Shift data by lag
    data_lagged = np.roll(data, -lag)
    data_lagged = data_lagged[:-lag] if lag > 0 else data_lagged
    data = data[:len(data_lagged)]

    # Default arguments for gnostic functions
    FLUSH = False
    VERBOSE = False

    if case == 'i':
        logger.info("Using Estimation Global Distribution Function (EGDF) for irrelevance computation.")
        # EGDF
        egdf_data = EGDF(flush=FLUSH, verbose=VERBOSE)
        egdf_data.fit(data)

        egdf_data_lagged = EGDF(flush=FLUSH, verbose=VERBOSE)
        egdf_data_lagged.fit(data_lagged)

        # Data Homogeneity
        logger.info("Performing data homogeneity check.")
        dh_data = DataHomogeneity(gdf=egdf_data, verbose=VERBOSE, flush=FLUSH)
        is_homo_data = dh_data.fit()

        dh_data_lagged = DataHomogeneity(gdf=egdf_data_lagged, verbose=VERBOSE, flush=FLUSH)
        is_homo_data_lagged = dh_data_lagged.fit()

        # data homogeneity check
        if not is_homo_data:
            logger.warning("Data is not homogeneous. Switching to S=1 for better results.")
            logger.info("Fitting EGDF with S=1.")
            egdf_data = EGDF(flush=FLUSH, verbose=VERBOSE, S=1)
            egdf_data.fit(data)
        
        if not is_homo_data_lagged:
            logger.warning("Lagged data is not homogeneous. Switching to S=1 for better results.")
            logger.info("Fitting EGDF with S=1.")
            egdf_data_lagged = EGDF(flush=FLUSH, verbose=VERBOSE, S=1)
            egdf_data_lagged.fit(data_lagged)

        # Get irrelevance of the data sample
        logger.info("Getting irrelevance of the data sample.")
        hc_data = np.mean(egdf_data.hi, axis=0)
        hc_data_lagged = np.mean(egdf_data_lagged.hi, axis=0)

    if case == 'j':
        logger.info("Using Quantifying Global Distribution Function (QGDF) for irrelevance computation.")
        # EGDF
        egdf_data = EGDF(flush=FLUSH, verbose=VERBOSE)
        egdf_data.fit(data)

        egdf_data_lagged = EGDF(flush=FLUSH, verbose=VERBOSE)
        egdf_data_lagged.fit(data_lagged)

        # Data Homogeneity
        logger.info("Performing data homogeneity check.")
        dh_data = DataHomogeneity(gdf=egdf_data, verbose=VERBOSE, flush=FLUSH)
        is_homo_data = dh_data.fit()

        dh_data_lagged = DataHomogeneity(gdf=egdf_data_lagged, verbose=VERBOSE, flush=FLUSH)
        is_homo_data_lagged = dh_data_lagged.fit()

        # data homogeneity check
        if not is_homo_data:
            logger.info("Data is not homogeneous.")
        if not is_homo_data_lagged:
            logger.info("Lagged data is not homogeneous.")

        # QGDF
        logger.info("Fitting QGDF with S=1.")
        qgdf_data = QGDF(flush=FLUSH, verbose=VERBOSE)
        qgdf_data.fit(data)

        qgdf_data_lagged = QGDF(flush=FLUSH, verbose=VERBOSE)
        qgdf_data_lagged.fit(data_lagged)

        # Get irrelevance of the data sample
        hc_data = np.mean(qgdf_data.hj, axis=0)
        hc_data_lagged = np.mean(qgdf_data_lagged.hj, axis=0)

        # Stop overflow by limiting big value in hc up to 1e12
        hc_data = np.clip(hc_data, 1, 1e12)
        hc_data_lagged = np.clip(hc_data_lagged, 1, 1e12)

        

    # Compute auto-covariance
    numerator = np.sum(hc_data * hc_data_lagged)
    denominator = (len(data) - lag)
    if denominator == 0:
        auto_covar = 0
    if denominator != 0:
        auto_covar = numerator / denominator

    return auto_covar