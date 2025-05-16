import numpy as np

def mean_squared_error(y_true, y_pred):
    """
    Computes the mean squared error.
    
    Returns
    -------
    float
        Average of squared differences between actual and predicted values.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)
