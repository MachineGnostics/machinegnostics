import numpy as np

def root_mean_squared_error(y_true, y_pred):
    """
    Computes the root mean squared error.
    
    Returns
    -------
    float
        Square root of the average of squared errors.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
