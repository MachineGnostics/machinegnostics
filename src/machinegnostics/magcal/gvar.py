'''
ManGo - Machine Gnostics Library
Copyright (C) 2025  ManGo Team

Author: Nirmal Parmar

Description: Calculates Gnostics Variance of the sample data Mc, c={i,j}
'''

import numpy as np
from machinegnostics.magcal.sample_characteristics import GnosticCharacteristicsSample

def gvariance(data:np.ndarray, case:str = 'i'):
    """
    Calculate the Gnostic variance (H) of a data sample.
    
    The Gnostic variance represents the spread of data through irrelevance functions
    and differs from classical statistical variance. It provides a robust measure
    of data dispersion that's less sensitive to outliers.
    
    Parameters
    ----------
    case : str, default='i'
        The type of variance to calculate:
        - 'i': Estimation irrelevance (Hi) - Used when estimating true values
              from noisy measurements
        - 'j': Quantification irrelevance (Hj) - Used when quantifying inherent
              data variability
    
    Returns
    -------
    float
        The calculated Gnostic variance value (H)
        Returns 0 if all data values are identical
    
    Notes
    -----
    This implementation follows the Gnostic theory where:
    1. Variance is based on irrelevance functions (Hi or Hj)
    2. Uses G-median as the location parameter
    3. Incorporates adaptive scaling through the scale parameter S
    
    Key differences from statistical variance:
    - Bounded and scale-invariant
    - Robust against outliers
    - Preserves data type characteristics
    - Different interpretation for estimation vs quantification cases

    Examples
    --------
    >>> data = np.array([1.0, 1.2, 0.8, 1.1, 0.9])
    >>> # Estimation variance (for noisy measurements)
    >>> var_i = gvar(data, case='i')
    >>> # Quantification variance (for inherent variability)
    >>> var_j = gvar(data, case='j')
    """
    gcs = GnosticCharacteristicsSample(data=data)
    gvariance = gcs._gnostic_variance(case=case, data=data)
    return gvariance