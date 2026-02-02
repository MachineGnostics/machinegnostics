import numpy as np

def make_anscombe_check_data(dataset_id=1):
    """
    Retrieves one of the four datasets from Anscombe's Quartet.
    
    Anscombe's quartet comprises four datasets that have nearly identical simple 
    descriptive statistics (mean, variance, correlation, regression line), yet 
    have very different distributions and appear very different when graphed.
    Ideally suited for demonstrating robust regression/gnostic models.

    Parameters
    ----------
    dataset_id : int, optional
        The identifier of the dataset to retrieve (1, 2, 3, or 4).
        1. Simple linear relationship with some noise.
        2. Non-linear relationship (polynomial).
        3. Linear relationship with a single outlier.
        4. Vertical line with one influential outlier.
        Default is 1.

    Returns
    -------
    X : numpy.ndarray
        The input feature array of shape (11,).
    y : numpy.ndarray
        The target array of shape (11,).

    Raises
    ------
    ValueError
        If dataset_id is not 1, 2, 3, or 4.

    Example
    -------
    >>> from machinegnostics.data.anscombe_data import make_anscombe_check_data
    >>> X, y = make_anscombe_check_data(dataset_id=3)
    >>> print(f"X mean: {np.mean(X):.2f}, y mean: {np.mean(y):.2f}")
    X mean: 9.00, y mean: 7.50
    """
    
    # Common X for datasets 1, 2, 3
    x_123 = np.array([10., 8., 13., 9., 11., 14., 6., 4., 12., 7., 5.])
    
    if dataset_id == 1:
        X = x_123
        y = np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68])
        
    elif dataset_id == 2:
        X = x_123
        y = np.array([9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74])
        
    elif dataset_id == 3:
        X = x_123
        y = np.array([7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73])
        
    elif dataset_id == 4:
        X = np.array([8., 8., 8., 8., 8., 8., 8., 19., 8., 8., 8.])
        y = np.array([6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89])
        
    else:
        raise ValueError("dataset_id must be 1, 2, 3, or 4.")
        
    return X, y
