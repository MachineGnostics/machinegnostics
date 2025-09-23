"""
Auto-Correlation Metric

This module provides a function to compute the auto-correlation of a data sample.

Author: Nirmal Parmar
Machine Gnostics
"""

import logging
from machinegnostics.magcal.util.logging import get_logger
import numpy as np
from machinegnostics.magcal import EGDF, QGDF, DataHomogeneity

def auto_correlation(data: np.ndarray, lag: int = 0, case: str = 'i', verbose: bool = False) -> float:
    """
    Calculate the Gnostic auto-correlation of a data sample.

    Auto-correlation measures the similarity between a data sample and a lagged version of itself.
    This function uses the principles of gnostic theory to compute robust estimates of auto-correlation.

    Parameters:
    ----------
    data : np.ndarray
        The data sample. Must be a 1D numpy array without NaN or Inf values.
    lag : int, optional, default=0
        The lag value for which the auto-correlation is computed. Must be non-negative and less than the length of the data.
    case : str, optional, default='i'
        Specifies the type of geometry to use:
        - 'i': Estimation geometry (EGDF).
        - 'j': Quantifying geometry (QGDF).
    verbose : bool, optional, default=False
        If True, enables detailed logging for debugging purposes.

    Returns:
    -------
    float
        The Gnostic auto-correlation coefficient for the given lag.

    Raises:
    ------
    ValueError
        If the input array is empty, contains NaN/Inf values, is not 1D, or if the lag is invalid.

    Examples:
    ---------
    Example 1: Compute auto-correlation for a simple dataset
    >>> import numpy as np
    >>> from machinegnostics.metrics import auto_correlation
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> lag = 1
    >>> auto_corr = auto_correlation(data, lag=lag, case='i', verbose=False)
    >>> print(f"Auto-Correlation (lag={lag}, case='i'): {auto_corr}")

    Notes:
    -----
    - This metric is robust to data uncertainty and provides meaningful estimates even in the presence of noise or outliers.
    - Ensure that the input data is preprocessed and cleaned for optimal results.
    """
    logger = get_logger('auto_correlation', level=logging.WARNING if not verbose else logging.INFO)
    logger.info("Starting auto-correlation computation.")

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

    # Compute correlation
    def compute_correlation(hc_data_1: np.ndarray, hc_data_2: np.ndarray) -> float:
        logger.debug("Computing correlation.")
        numerator = np.sum(hc_data_1 * hc_data_2)
        denominator = (np.sqrt(np.sum(hc_data_1**2)) * np.sqrt(np.sum(hc_data_2**2))) 
        corr = numerator / denominator
        if denominator == 0:
            return np.nan
        return corr
    
    auto_corr = compute_correlation(hc_data, hc_data_lagged)

    return auto_corr