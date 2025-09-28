import numpy as np

def np_max_float():
    """
    Returns the maximum float value that can be represented in NumPy.
    
    Returns
    -------
    float
        The maximum float value.
    """
    return np.finfo(float).max

def np_min_float():
    """
    Returns the minimum float value that can be represented in NumPy.
    
    Returns
    -------
    float
        The minimum float value.
    """
    return np.finfo(float).min

def np_eps_float():
    """
    Returns the smallest positive float value that can be represented in NumPy.
    
    Returns
    -------
    float
        The smallest positive float value.
    """
    return np.finfo(float).eps