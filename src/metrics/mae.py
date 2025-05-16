import numpy as np

def mean_absolute_error(y_true, y_pred):
    """
    Computes the mean absolute error.
    
    Returns
    -------
    float
        Average absolute difference between actual and predicted values.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))
