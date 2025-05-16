import numpy as np

def r2_score(y_true, y_pred):
    """
    Computes the coefficient of determination (R² score).
    
    Returns
    -------
    float
        Proportion of variance explained (1 is perfect prediction).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def adjusted_r2_score(y_true, y_pred, n_features):
    """
    Computes the adjusted R² score.
    
    Parameters
    ----------
    n_features : int
        Number of features (independent variables) in the model.
    
    Returns
    -------
    float
        Adjusted R² accounting for number of predictors.
    """
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)