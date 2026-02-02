import numpy as np

def make_regression_check_data(n_samples=20, slope=3.5, intercept=10.0, noise_level=2.0, seed=42):
    """
    Generates synthetic regression data for validating Machine Gnostics models.

    This function creates a simple linear dataset (y = slope * x + intercept + noise)
    that serves as a 'hello world' for checking if a regression model is working 
    correctly. It ensures reproducibility via a fixed seed.

    Parameters
    ----------
    n_samples : int, optional
        The number of data points to generate. Default is 20.
    slope : float, optional
        The true coefficient (slope) of the underlying linear relationship. 
        Default is 3.5.
    intercept : float, optional
        The true intercept (bias) of the underlying linear relationship. 
        Default is 10.0.
    noise_level : float, optional
        The standard deviation of the Gaussian noise added to the target variable. 
        Controls how 'noisy' the data is. Default is 2.0.
    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    X : numpy.ndarray
        The input feature array of shape (n_samples,).
        Values are uniformly distributed between 0 and 10.
    y : numpy.ndarray
        The target array of shape (n_samples,).

    Example
    -------
    >>> from machinegnostics.data.reg_data import make_regression_check_data
    >>> X, y = make_regression_check_data(n_samples=50)
    >>> print(f"X shape: {X.shape}, y shape: {y.shape}")
    X shape: (50,), y shape: (50,)
    """
    rng = np.random.default_rng(seed)
    
    # Generate X values between 0 and 10
    X = 10 * rng.random(n_samples)
    
    # Generate noise
    noise = rng.normal(loc=0, scale=noise_level, size=n_samples)
    
    # Generate y = mx + c + noise
    y = slope * X + intercept + noise
    
    return X, y
