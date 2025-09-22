'''
ManGo - Machine Gnostics Library
Copyright (C) 2025  ManGo Team

Author: Nirmal Parmar

Description: Implementation of Gnostic Median calculations
'''

import numpy as np
from machinegnostics.magcal.sample_characteristics import GnosticCharacteristicsSample

def gmedian(data, case='i', z_range=None, tol=1e-8):
    """
    Calculate the Gnostic Characteristics (Modulus, Median, Correlation, Auto-correlation, etc.) of a data sample.
    
    The G-median is defined as the value Z_med for which the sum of irrelevances equals zero.
    Implements both quantifying and estimating cases based on equations 14.23 and 14.24.
    
    Notes
    -----
    The Gnostic median fundamentally differs from traditional statistical measures like the 
    arithmetic mean and statistical median in its approach to data analysis. While the 
    arithmetic mean assumes symmetric error distribution and is highly sensitive to outliers, 
    the G-median makes no such distributional assumptions and remains robust against outliers 
    through its irrelevance weighting mechanism. Similarly, where the statistical median only 
    considers data ordering while ignoring magnitude differences, the G-median accounts for 
    both order and magnitude in its calculations. Furthermore, the G-median offers two distinct 
    variants - quantifying and estimating - suited for different analytical purposes, and 
    incorporates data reliability through sophisticated irrelevance measures. This makes it 
    particularly valuable for analyzing real-world data where traditional statistical 
    assumptions may not hold and where data quality varies across observations.
    
    Parameters
    ----------
    data : array-like
        Input data sample
    case : str, default='i'
        The type of G-median to calculate:
        - 'j' means quantifying case
        - 'i' means estimating case (default)
    z_range : tuple, optional
        Initial search range for Z_med (min, max). If None, will be determined from data
    tol : float, default=1e-8
        Tolerance for convergence
        
    Returns
    -------
    float
        The calculated G-median value

    Examples
    --------
    >>> import numpy as np
    >>> from machinegnostics.magcal.gmed import gmedian
    >>> data = np.random.rand(100)
    >>> gmedian(data, case='i')
        
    References
    ----------
    .. [1] Kovanic P., Humber M.B (2015) The Economics of Information - Mathematical
           Gnostics for Data Analysis. http://www.math-gnostics.eu/books/
    """
    # Convert input to numpy array and validate
    data = np.asarray(data, dtype=np.float64)
    
    # Check for empty or invalid input
    if data.size == 0:
        raise ValueError("Input data array is empty")
    
    if np.any(~np.isfinite(data)):
        raise ValueError("Input data contains NaN or infinite values")
    
    # Validate case parameter
    if case not in ['i', 'j']:
        raise ValueError("case must be either 'i' (quantifying) or 'j' (estimating)")

    # Initialize z_range with progressive widening strategy
    z_min, z_max = np.min(data), np.max(data)
    if z_min == z_max:
        return z_max
    
    range_buffers = [0.1, 0.25, 0.5]  # Progressive buffer sizes
    
    for attempt, buffer in enumerate(range_buffers, 1):
        try:
            if z_range is None:
                # Calculate range with current buffer
                range_width = z_max - z_min
                current_z_range = (
                    z_min - range_width * buffer,
                    z_max + range_width * buffer
                )
            else:
                current_z_range = z_range
            
            # Initialize LocationParameter
            lp = GnosticCharacteristicsSample(data=data, tol=tol)
            
            # Attempt G-median calculation
            result = lp._gnostic_median(case=case, z_range=current_z_range)
            
            # Check convergence
            if result.converged:
                return result.root
            elif attempt == len(range_buffers):
                raise RuntimeError(
                    "G-median calculation failed to converge after multiple attempts "
                    f"with different range buffers. Last range tried: {current_z_range}"
                )
                
        except Exception as e:
            if attempt == len(range_buffers):
                raise RuntimeError(
                    f"Error calculating G-median after {attempt} attempts: {str(e)}\n"
                    f"Try providing explicit z_range or adjusting tolerance."
                )
            continue
            
    raise RuntimeError("Unexpected failure in G-median calculation")