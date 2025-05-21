'''
ManGo - Machine Gnostics Library
Copyright (C) 2025  ManGo Team

This work is licensed under the terms of the GNU General Public License version 3.0.
For more details, see <https://www.gnu.org/licenses/gpl-3.0.html>.
'''

import numpy as np
from machinegnostics.magcal.sample_characteristics import GnosticCharacteristicsSample

def gautocovariance(data: np.ndarray, case: str = 'i', K: int = 1):
    """
    Compute the gnostic autocovariance for a given data sample.
    
    Gnostic autocovariance measures the correlation between elements in a data sample 
    separated by a lag K, using a specific gnostic irrelevance function. 
    This function provides a generalized autocovariance based on the irrelevance (Hi or Hj)
    of the data, as per the theory described in equation 14.19.
    
    Parameters
    ----------
    data : np.ndarray
        Input data sample. Should be a 1D array of numerical values representing the observations.
    case : str, optional
        Specifies the type of irrelevance function to use for autocovariance calculation.
        - 'i': Estimation case using Hi irrelevance (default)
        - 'j': Quantification case using Hj irrelevance
    K : int, optional
        Lag parameter specifying the separation between data points to correlate.
        Must be an integer in the range [1, N-1], where N is the length of the data sample.
        Default is 1.
    
    Returns
    -------
    float
        The computed gnostic autocovariance value for the given lag and irrelevance case.
    
    Raises
    ------
    ValueError
        If K is not in the valid range [1, N-1].
        If `case` is not one of {'i', 'j'}.
        If `data` is not a 1D numpy array.
       
    where:
        - N is the sample size (length of `data`)
        - K is the lag parameter
        - h_c is the irrelevance function (Hi for 'i', Hj for 'j')
        - Omega_i are the data angles (domain-specific transformation, see implementation)
       
    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.rand(100)
    >>> gautocovariance(data, case='i', K=1)
    0.0024
    
    >>> gautocovariance(data, case='j', K=2)
    -0.014
    
    """
    gcs = GnosticCharacteristicsSample(data=data)
    gnostic_acor = gcs._gnostic_autocovariance(K=K, case=case)
    return gnostic_acor