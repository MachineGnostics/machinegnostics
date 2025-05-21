'''
ManGo - Machine Gnostics Library
Copyright (C) 2025  ManGo Team

This work is licensed under the terms of the GNU General Public License version 3.0.
For more details, see <https://www.gnu.org/licenses/gpl-3.0.html>.
'''

import numpy as np
from machinegnostics.magcal.sample_characteristics import GnosticCharacteristicsSample

def gcovariance(data_1: np.ndarray, data_2: np.ndarray, case: str = 'i'):
    """
    Compute the Gnostic crosscovariance between two data samples.

    Gnostic crosscovariance measures the correlation between two separate data samples
    using irrelevance functions, providing a robust alternative to classical crosscovariance.
    This method is based on Gnostic theory (see equation 14.20) and leverages irrelevance
    functions (Hi or Hj) to suppress the influence of outliers and adapt to the underlying
    data structure.

    Parameters
    ----------
    data_1 : np.ndarray
        First data sample (array-like, 1D) for crosscovariance calculation.
    data_2 : np.ndarray
        Second data sample (array-like, 1D) to be compared with `data_1`.
        Must be of the same length as `data_1`.
    case : str, default='i'
        Type of irrelevance function to use:
        - 'i': Estimation irrelevance (Hi), suitable for estimating true values in the presence of noise or outliers.
        - 'j': Quantification irrelevance (Hj), suitable for quantifying inherent variability between the samples.

    Returns
    -------
    float
        The computed Gnostic crosscovariance value.
        Returns 0.0 if both data samples are constant or perfectly uncorrelated under the irrelevance mapping.

    Raises
    ------
    ValueError
        If the input arrays `data_1` and `data_2` do not have the same length.
        If either input is not a 1D numpy array.
        If `case` is not one of {'i', 'j'}.

    where:
        - N is the sample size (length of both data arrays)
        - h_c is the irrelevance function (Hi for 'i', Hj for 'j')
        - Omega_{n,A} and Omega_{n,B} are the gnostic angles (domain-specific transformation) from samples A and B
        - The sum is taken over all sample points n

    Unlike classical crosscovariance:
        - Gnostic crosscovariance is bounded and robust to outliers.
        - It remains meaningful for non-Gaussian, heavy-tailed, or contaminated data.
        - The choice of 'case' changes the interpretation (estimation vs. quantification).

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1.0, 2.0, 3.0, 4.0])
    >>> y = np.array([1.1, 1.9, 3.2, 3.8])
    >>> # Robust estimation crosscovariance
    >>> gcovariance(x, y, case='i')
    0.121
    >>> # Quantification crosscovariance
    >>> gcovariance(x, y, case='j')
    0.107

    """
    gcs = GnosticCharacteristicsSample(data=data_1)
    gcov = gcs._gnostic_crosscovariance(other_data=data_2, case=case)
    return gcov