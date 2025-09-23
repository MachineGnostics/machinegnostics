'''
Gnostic Correlation Metric

This module provides a function to compute the Gnostic correlation between two data samples.

Author: Nirmal Parmar
Machine Gnostics
'''

import numpy as np
from machinegnostics.magcal import EGDF, QGDF, DataHomogeneity

def correlation(data_1: np.ndarray, data_2: np.ndarray, case: str = 'i') -> float:
    """
    Calculate the Gnostic correlation coefficient between two data samples.

    The Gnostic correlation metric is based on the principles of gnostic theory, which
    provides robust estimates of data correlations. This metric leverages the concepts
    of estimating irrelevances and quantifying irrelevances, which are robust measures
    of data uncertainty. These irrelevances are aggregated differently:

    - Quantifying irrelevances are aggregated additively as hyperbolic sines.
    - Estimating irrelevances are aggregated as trigonometric sines.

    Both types of irrelevances converge to linear errors of observed data in cases of
    weak uncertainty. The product of quantifying irrelevances and estimating irrelevances
    serves as a generalization of data products. Normalized estimates of their means
    provide robust estimates of correlations.

    This function computes the Gnostic correlation coefficient using the following steps:
    1. Validate the input data for consistency, shape, and content.
    2. Depending on the `case` parameter:
       - For `case='i'` (estimating geometry):
         - Use Estimation Geometry Distribution Functions (EGDF) to compute estimating irrelevances.
         - Assess data homogeneity using DataHomogeneity.
       - For `case='j'` (quantifying geometry):
         - Use Quantifying Geometry Distribution Functions (QGDF) to compute quantifying irrelevances.
         - Clip extreme values to prevent overflow.
         - Assess data homogeneity using EGDF and DataHomogeneity.
    3. Compute the correlation coefficient as the normalized product of irrelevances.

    Parameters:
    ----------
    data_1 : np.ndarray
        The first data sample. Must be a 1D numpy array without NaN or Inf values.
    data_2 : np.ndarray
        The second data sample. Must be a 1D numpy array without NaN or Inf values.
    case : str, optional, default='i'
        Specifies the type of geometry to use:
        - 'i': Estimation geometry (EGDF).
        - 'j': Quantifying geometry (QGDF).

    Returns:
    -------
    float
        The Gnostic correlation coefficient between the two data samples.

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

    # Validate inputs
    if len(data_1) != len(data_2):
        raise ValueError("Input arrays must have the same length.")
    if len(data_1) == 0 or len(data_2) == 0:
        raise ValueError("Input arrays must not be empty.")
    if not isinstance(data_1, np.ndarray) or not isinstance(data_2, np.ndarray):
        raise ValueError("Inputs must be numpy arrays.")
    # flatten the arrays if they are not 1D
    data_1 = data_1.flatten()
    data_2 = data_2.flatten()
    if data_1.ndim != 1 or data_2.ndim != 1:
        raise ValueError("Input arrays must be 1D.")
    # avoid inf and nan in data
    if np.any(np.isnan(data_1)) or np.any(np.isnan(data_2)):
        raise ValueError("Input arrays must not contain NaN values.")
    if np.any(np.isinf(data_1)) or np.any(np.isinf(data_2)):
        raise ValueError("Input arrays must not contain Inf values.")
    if case not in ['i', 'j']:
        raise ValueError("Case must be 'i' for estimation geometry or 'j' for quantifying geometry.")
    
    # default arg
    FLUSH = False
    VERBOSE = False
    
    if case == 'i':
        # EGDF 
        egdf_data_1 = EGDF(flush=FLUSH, verbose=VERBOSE)
        egdf_data_1.fit(data_1)

        egdf_data_2 = EGDF(flush=FLUSH, verbose=VERBOSE)
        egdf_data_2.fit(data_2)

        # Data Homogeneity
        dh_data_1 = DataHomogeneity(gdf=egdf_data_1, verbose=VERBOSE, flush=FLUSH)
        is_homo_data_1 = dh_data_1.fit()

        dh_data_2 = DataHomogeneity(gdf=egdf_data_2, verbose=VERBOSE, flush=FLUSH)
        is_homo_data_2 = dh_data_2.fit()

        # get irrelevance of the data sample
        hc_data_1 = np.mean(egdf_data_1.hi, axis=0)
        hc_data_2 = np.mean(egdf_data_2.hi, axis=0)

    if case == 'j':
        # QGDF
        qgdf_data_1 = QGDF(flush=FLUSH, verbose=VERBOSE)
        qgdf_data_1.fit(data_1)

        qgdf_data_2 = QGDF(flush=FLUSH, verbose=VERBOSE)
        qgdf_data_2.fit(data_2)

        # get irrelevance of the data sample
        hc_data_1 = np.mean(qgdf_data_1.hi, axis=0)
        hc_data_2 = np.mean(qgdf_data_2.hi, axis=0)

        # stop overflow by limiting big value in hc up to 1e12
        hc_data_1 = np.clip(hc_data_1, 1, 1e12)
        hc_data_2 = np.clip(hc_data_2, 1, 1e12)

        # EGDF 
        egdf_data_1 = EGDF(flush=FLUSH, verbose=VERBOSE)
        egdf_data_1.fit(data_1)

        egdf_data_2 = EGDF(flush=FLUSH, verbose=VERBOSE)
        egdf_data_2.fit(data_2)

        # data homogeneity
        dh_data_1 = DataHomogeneity(gdf=egdf_data_1, verbose=VERBOSE, flush=FLUSH)
        is_homo_data_1 = dh_data_1.fit()

        dh_data_2 = DataHomogeneity(gdf=egdf_data_2, verbose=VERBOSE, flush=FLUSH)
        is_homo_data_2 = dh_data_2.fit()

    # raise warning if data is not homogeneous
    if not is_homo_data_1:
        print("Warning: Data 1 is not homogeneous. Please check the data distribution. For better results, use scale parameter as 1.")
        egdf_data_1 = EGDF(flush=FLUSH, verbose=VERBOSE, S=1)
        egdf_data_1.fit(data_1)
    if not is_homo_data_2:
        print("Warning: Data 2 is not homogeneous. Please check the data distribution. For better results, use scale parameter as 1.")
        egdf_data_2 = EGDF(flush=FLUSH, verbose=VERBOSE, S=1)
        egdf_data_2.fit(data_2)

    # Compute correlation
    def compute_correlation(hc_data_1: np.ndarray, hc_data_2: np.ndarray) -> float:
        numerator = np.sum(hc_data_1 * hc_data_2)
        denominator = (np.sqrt(np.sum(hc_data_1**2)) * np.sqrt(np.sum(hc_data_2**2))) 
        corr = numerator / denominator
        if denominator == 0:
            return np.nan
        return corr
    
    # Compute correlation
    corr = compute_correlation(hc_data_1, hc_data_2)
    return corr
