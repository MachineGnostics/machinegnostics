'''
ManGo - Machine Gnostics Library
Copyright (C) 2025  ManGo Team

This work is licensed under the terms of the GNU General Public License version 3.0.
'''

import numpy as np
from src.magcal.characteristics import GnosticsCharacteristics

def hc(y_true, y_pred, case:str='i'):
    """
    Calculate the Gnostic Characteristics (Hc) metric of the data sample.

    i  - for estimating gnostic relevance
    j  - for estimating gnostic irrelevance

    The HC metric is used to evaluate the performance of a model by comparing
    the predicted values with the true values's relevance or irrelevance.
    It is calculated as the sum of the characteristics of the model.

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.
    case : str, optional
        Case to be used for calculation. Options are 'i' or 'j'. Default is 'i'.

    Returns
    -------
    float
        The calculated HC value.

    Example
    -------
    >>> y_true = [1, 2, 3]
    >>> y_pred = [1, 2, 3]
    >>> from mango.metrics import hc
    >>> hc_value = hc(y_true, y_pred, case='i')
    >>> print(hc_value)
    """
    
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Check if the lengths of y_true and y_pred match
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    # Calculate the ratio R = Z / Z0
    R = y_true / y_pred

    # Create an instance of GnosticsCharacteristics
    gnostics = GnosticsCharacteristics(R=R)

    # Calculate q and q1
    q, q1 = gnostics._get_q_q1()

    # Calculate fi, fj, hi, hj based on the case
    if case == 'i':
        hc = gnostics._hi(q, q1)
    
    elif case == 'j':
        hc = gnostics._hj(q, q1)
    
    else:
        raise ValueError("Invalid case. Use 'i' or 'j'.")

    return np.sum(hc)