"""
Auto-Correlation Metric

This module provides a function to compute the auto-correlation of a data sample.

Author: Nirmal Parmar
Machine Gnostics
"""

import numpy as np
from machinegnostics.magcal import EGDF, QGDF, DataHomogeneity

def auto_correlation(data: np.ndarray, lag: int = 0, case: str = 'i') -> float:
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

    Returns:
    -------
    float
        The Gnostic auto-correlation coefficient for the given lag.

    Raises:
    ------
    ValueError
        If the input array is empty, contains NaN/Inf values, is not 1D, or if the lag is invalid.

    Notes:
    -----
    - This metric is robust to data uncertainty and provides meaningful estimates even in the presence of noise or outliers.
    - Ensure that the input data is preprocessed and cleaned for optimal results.
    """

    # Validate inputs
    if not isinstance(data, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    # flatten data
    data = data.flatten()
    if data.ndim != 1:
        raise ValueError("Input array must be 1D.")
    if len(data) == 0:
        raise ValueError("Input array must not be empty.")
    if np.any(np.isnan(data)):
        raise ValueError("Input array must not contain NaN values.")
    if np.any(np.isinf(data)):
        raise ValueError("Input array must not contain Inf values.")
    if lag < 0 or lag >= len(data):
        raise ValueError("Lag must be non-negative and less than the length of the data.")
    if case not in ['i', 'j']:
        raise ValueError("Case must be 'i' for estimation geometry or 'j' for quantifying geometry.")

    # Shift data by lag
    data_lagged = np.roll(data, -lag)
    data_lagged = data_lagged[:-lag] if lag > 0 else data_lagged
    data = data[:len(data_lagged)]

    # Default arguments for gnostic functions
    FLUSH = False
    verbose = False

    if case == 'i':
        # EGDF
        egdf_data = EGDF(flush=FLUSH, verbose=verbose)
        egdf_data.fit(data)

        egdf_data_lagged = EGDF(flush=FLUSH, verbose=verbose)
        egdf_data_lagged.fit(data_lagged)

        # Data Homogeneity
        dh_data = DataHomogeneity(gdf=egdf_data, verbose=verbose, flush=FLUSH)
        is_homo_data = dh_data.fit()

        dh_data_lagged = DataHomogeneity(gdf=egdf_data_lagged, verbose=verbose, flush=FLUSH)
        is_homo_data_lagged = dh_data_lagged.fit()

        # Get irrelevance of the data sample
        hc_data = np.mean(egdf_data.hi, axis=0)
        hc_data_lagged = np.mean(egdf_data_lagged.hi, axis=0)

    if case == 'j':
        # QGDF
        qgdf_data = QGDF(flush=FLUSH, verbose=verbose)
        qgdf_data.fit(data)

        qgdf_data_lagged = QGDF(flush=FLUSH, verbose=verbose)
        qgdf_data_lagged.fit(data_lagged)

        # Get irrelevance of the data sample
        hc_data = np.mean(qgdf_data.hi, axis=0)
        hc_data_lagged = np.mean(qgdf_data_lagged.hi, axis=0)

        # Stop overflow by limiting big value in hc up to 1e12
        hc_data = np.clip(hc_data, 1, 1e12)
        hc_data_lagged = np.clip(hc_data_lagged, 1, 1e12)

    # Compute correlation
    def compute_correlation(hc_data_1: np.ndarray, hc_data_2: np.ndarray) -> float:
        numerator = np.sum(hc_data_1 * hc_data_2)
        denominator = (np.sqrt(np.sum(hc_data_1**2)) * np.sqrt(np.sum(hc_data_2**2))) 
        corr = numerator / denominator
        if denominator == 0:
            return np.nan
        return corr
    auto_corr = compute_correlation(hc_data, hc_data_lagged)

    return auto_corr