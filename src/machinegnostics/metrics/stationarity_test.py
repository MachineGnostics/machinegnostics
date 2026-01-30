'''
Stationarity Test - Machine Gnostics Framework

Machine Gnostics

Author: Nirmal Parmar
'''

import logging
import numpy as np
from typing import Union, List
from machinegnostics.magcal.util.logging import get_logger
from machinegnostics.magcal import EGDF, DataHomogeneity

def stationarity_test(data: Union[np.ndarray, List], 
                      window_size: int = 10,
                      S: str = 'auto',
                      data_form: str = 'a',
                      verbose: bool = False) -> bool:
    """
    Check for stationarity in a time series using Residual Entropy homogeneity.
    
    This function analyzes the stationarity of a time series by computing the 
    Residual Entropy (RE) over a sliding window. It then determines if the 
    sequence of RE values is homogeneous using the DataHomogeneity test. 
    If the RE sequence is homogeneous, the time series is considered stationary.
    
    Parameters:
    -----------
    data : array-like of shape (n_samples,)
        Time series data to analyze.
    window_size : int, optional (default=10)
        The size of the sliding window used to calculate Residual Entropy.
        Must be less than the length of the data.
    S : str, optional (default='auto')
        Scale parameter for EGDF fitting. 'auto' lets the algorithm choose.
    data_form : str, optional (default='a')
        Form of the input data: 'a' for additive, 'm' for multiplicative.
    verbose : bool, optional (default=False)
        If True, enables detailed logging for debugging purposes.
        
    Returns:
    --------
    bool
        True if the time series is stationary (Residual Entropy is homogeneous),
        False otherwise.
        
    Raises:
    -------
    TypeError
        If input data is not array-like.
    ValueError
        If input dimensions are incorrect or window_size is invalid.
        
    Example:
    --------
    >>> from machinegnostics.metrics import stationarity_test
    >>> import numpy as np
    >>> data = np.random.normal(0, 1, 100)
    >>> is_stationary = stationarity_test(data, window_size=20)
    >>> print(f"Is stationary: {is_stationary}")
    """
    logger = get_logger('stationarity_test', level=logging.WARNING if not verbose else logging.INFO)
    logger.info("Starting stationarity test...")

    # Validate inputs
    if not isinstance(data, (list, tuple, np.ndarray)):
        logger.error("data must be array-like (list, tuple, or numpy array).")
        raise TypeError("data must be array-like (list, tuple, or numpy array).")
    
    # Convert to numpy array
    data = np.asarray(data).flatten()
    n_samples = len(data)
    
    # Check dimensions and window size
    if n_samples == 0:
        logger.error("Input data is empty.")
        raise ValueError("Input data is empty.")
    
    if window_size >= n_samples:
        logger.error(f"window_size ({window_size}) must be less than the length of data ({n_samples}).")
        raise ValueError(f"window_size ({window_size}) must be less than the length of data ({n_samples}).")
        
    if window_size < 3: # EGDF might need a few points
        logger.warning(f"window_size ({window_size}) is very small. Results may be unstable.")

    logger.info(f"Processing {n_samples} samples with sliding window of size {window_size}...")
    
    re_values = []
    
    # Sliding window loop
    # We loop such that we can extract full windows of size `window_size`
    for i in range(n_samples - window_size + 1):
        window = data[i : i + window_size]
        
        try:
            # Fit EGDF on the window
            # Suppress internal EGDF logging unless very verbose (not passed here usually)
            egdf = EGDF(verbose=False, 
                        S=S, 
                        data_form=data_form, 
                        n_points=100)
            egdf.fit(window)
            
            # Extract Residual Entropy
            re = egdf.params.get('residual_entropy', np.nan)
            re_values.append(re)
            
        except Exception as e:
            logger.debug(f"EGDF fit failed at index {i}: {e}")
            # Append NaN to maintain loose temporal alignment if needed, 
            # though we filter NaNs later for homogeneity check.
            re_values.append(np.nan)

    re_values = np.array(re_values)
    
    # Filter out NaNs (failed fits or undefined entropy)
    valid_re_values = re_values[~np.isnan(re_values)]
    n_valid = len(valid_re_values)

    logger.info(f"Computed {n_valid} valid Residual Entropy values.")
    
    if n_valid < 10: # Heuristic minimum points for homogeneity check
        logger.warning("Too few valid Residual Entropy values extracted to perform homogeneity test.")
        # If we can't test, we probably can't claim stationarity.
        return False

    try:
        logger.info("Performing homogeneity test on Residual Entropy values...")
        # Fit EGDF on the sequence of Residual Entropy values
        re_egdf = EGDF(verbose=False, S=S, data_form='a', n_points=100)
        re_egdf.fit(valid_re_values)
        
        # Check Homogeneity
        homo = DataHomogeneity(gdf=re_egdf)
        is_homogeneous = homo.fit()
        
        logger.info(f"Stationarity result: {is_homogeneous}")
        return is_homogeneous
        
    except Exception as e:
        logger.error(f"Homogeneity test failed: {e}")
        # Default to False if check fails
        return False
