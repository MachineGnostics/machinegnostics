'''
ManGo - Machine Gnostics Library
Copyright (C) 2025  ManGo Team

This work is licensed under the terms of the GNU General Public License version 3.0.
For more details, see <https://www.gnu.org/licenses/gpl-3.0.html>.
'''

import numpy as np
from machinegnostics.magcal.sample_characteristics import GnosticCharacteristicsSample

def gcorrelation(data_1:np.ndarray, data_2:np.ndarray, case:str = 'i'):
    """
    Calculate the Gnostic correlation between two data samples.
    
    Gnostic correlation measures the relationship between two data samples using
    irrelevance functions, providing a robust alternative to classical Pearson
    correlation. It is less sensitive to outliers and makes no assumptions about
    the underlying distribution of the data.

    If self.data is 2D (shape: [n_samples, n_features]), 
    computes correlation for each feature column with other_data.
    If self.data is 1D, computes correlation directly.

    Parameters
    ----------
    data_1 : np.ndarray
        First data sample
    data_2 : np.ndarray
        Second data sample to compare with data_1
        Must have the same length as data_1
    case : str, default='i'
        The type of correlation to calculate:
        - 'i': Estimation case - Used when data contains measurement errors
              Returns values in [-1, 1], similar to Pearson correlation
        - 'j': Quantification case - Used for inherent data variability
              Returns positive values, indicating strength of relationship
    
    Returns
    -------
    float
        The calculated Gnostic correlation coefficient
        For case='i': Range [-1, 1]
        For case='j': Range [0, ∞)
    
    Notes
    -----
    The Gnostic correlation:
    1. Is robust against outliers
    2. Does not assume normal distribution
    3. Handles both estimation and quantification scenarios
    4. Preserves data type characteristics
    5. Uses G-median as location parameter

    Key differences from Pearson correlation:
    - More robust to outliers
    - Two different interpretations (i/j cases)
    - No distributional assumptions
    - Better handles non-linear relationships
    
    Examples
    --------
    >>> # Linear relationship with noise
    >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> y = np.array([0.9, 2.1, 2.9, 4.2, 4.8])
    >>> gcor = gcorrelation(x, y, case='i')
    >>> print(f"Estimation correlation: {gcor:.3f}")
    >>> from machinegnostics.magcal.gcor import gcorrelation
    >>> # Inherent variability analysis
    >>> measurements_A = np.array([10.1, 10.3, 9.8, 10.2, 10.0])
    >>> measurements_B = np.array([5.1, 5.2, 4.9, 5.3, 5.0])
    >>> gcor = gcorrelation(measurements_A, measurements_B, case='j')
    >>> print(f"Quantification correlation: {gcor:.3f}")
    
    Raises
    ------
    ValueError
        If input arrays have different lengths
        If case is not 'i' or 'j'
        If input arrays are empty
        
    References
    ----------
    .. [1] Kovanic P., Humber M.B (2015) The Economics of Information - Mathematical
           Gnostics for Data Analysis, Chapter 14.3.3
    """
    gcs = GnosticCharacteristicsSample(data=data_1)
    cor = gcs._gnostic_correlation(other_data=data_2, case=case)
    return cor