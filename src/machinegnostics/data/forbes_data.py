import numpy as np

def make_forbes_check_data():
    """
    Retrieves Forbes' dataset (1857) on boiling points in the Alps.
    
    A classic dataset often used to demonstrate robust regression methods. 
    It consists of 17 observations of the boiling point of water and 
    barometric pressure at different locations in the Alps.
    
    It is well-known for measuring the relationship between pressure and 
    boiling point, usually fitting: 100 * log10(Pressure) vs BoilingPoint.
    Observation 12 is widely considered an outlier due to measurement error.

    Features (X):
        Boiling Point: Temperature in degrees Fahrenheit.

    Target (y):
        Pressure: Barometric pressure in inches of mercury.

    Returns
    -------
    X : numpy.ndarray
        The input feature array of shape (17, 1).
    y : numpy.ndarray
        The target array of shape (17,).

    Example
    -------
    >>> from machinegnostics.data.forbes_data import make_forbes_check_data
    >>> X, y = make_forbes_check_data()
    >>> print(f"X shape: {X.shape}, y shape: {y.shape}")
    X shape: (17, 1), y shape: (17,)
    """
    
    # Column 0: Boiling Point (F)
    # Column 1: Pressure (Inches Hg)
    data = np.array([
        [194.5, 20.79],
        [194.3, 20.79],
        [197.9, 22.40],
        [198.4, 22.67],
        [199.4, 23.15],
        [199.9, 23.35],
        [200.9, 23.89],
        [201.1, 23.99],
        [201.4, 24.02],
        [201.3, 24.01],
        [203.6, 25.14],
        [204.6, 26.57],
        [209.5, 28.49],
        [208.6, 27.76],
        [210.7, 29.04],
        [211.9, 29.88],
        [212.2, 30.06]
    ])

    X = data[:, 0:1]  # Keep 2D shape for features
    y = data[:, 1]

    return X, y
