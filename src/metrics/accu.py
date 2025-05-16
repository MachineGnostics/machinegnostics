import numpy as np

def accuracy_score(y_true, y_pred):
    """
    Computes the classification accuracy.
    
    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
    
    Returns
    -------
    float
        Accuracy (0 to 1).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)