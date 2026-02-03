import numpy as np

def make_regression_check_data(n_samples=20, slope=3.5, intercept=10.0, noise_level=2.0, degree=1, function_type='poly', outlier_ratio=0.0, seed=42):
    """
    Generates synthetic regression data for validating Machine Gnostics models.

    This function creates a simple linear, polynomial, or sinusoidal dataset that 
    serves as a 'hello world' for checking if a regression model is working 
    correctly. It ensures reproducibility via a fixed seed.

    Parameters
    ----------
    n_samples : int, optional
        The number of data points to generate. Default is 20.
    slope : float, optional
        The true coefficient (slope) or amplitude. 
        For linear/poly: The slope of the line.
        For sin/cos: The amplitude of the wave.
        Default is 3.5.
    intercept : float, optional
        The true intercept (bias) of the underlying relationship. 
        Default is 10.0.
    noise_level : float, optional
        The standard deviation of the Gaussian noise added to the target variable. 
        Controls how 'noisy' the data is. Default is 2.0.
    degree : int, optional
        The degree of the polynomial. Used only if function_type='poly'.
        Default is 1 (linear).
    function_type : str, optional
        The type of component function. Options: 'poly', 'sin', 'cos'.
        Default is 'poly'.
    outlier_ratio : float, optional
        The proportion of samples to contaminate with outliers (0.0 to 1.0).
        These points will have their y-values shifted significantly.
        Default is 0.0 (no outliers).
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
    >>> X, y = make_regression_check_data(n_samples=50, function_type='sin', outlier_ratio=0.1)
    >>> print(f"X shape: {X.shape}, y shape: {y.shape}")
    X shape: (50,), y shape: (50,)
    """
    rng = np.random.default_rng(seed)
    
    # Generate X values between 0 and 10
    X = 10 * rng.random(n_samples)
    
    # Generate noise
    noise = rng.normal(loc=0, scale=noise_level, size=n_samples)
    
    if function_type.lower() == 'sin':
        # slope acts as amplitude
        y = slope * np.sin(X) + intercept + noise
    elif function_type.lower() == 'cos':
        y = slope * np.cos(X) + intercept + noise
    else:
        # Generate y = mx + c + noise (plus higher order terms)
        y = np.full(n_samples, intercept) + noise
        
        for d in range(1, degree + 1):
            if d == 1:
                y += slope * X
            else:
                # Scale down higher order terms to maintain visualization friendly range
                # X is 0-10. X^2 is 0-100. dividing by 10 makes it 0-10 scale.
                term_scale = 10.0 ** (d - 1)
                y += slope * (X ** d) / term_scale

    # Add outliers
    if outlier_ratio > 0:
        n_outliers = int(n_samples * outlier_ratio)
        if n_outliers > 0:
            outlier_indices = rng.choice(n_samples, n_outliers, replace=False)
            # Add significant shift (5x to 10x the standard deviation of data or slope)
            # We use a random sign (+/-)
            shift_magnitude = max(slope * 5, 20.0) # Ensure at least shift of 20
            shifts = shift_magnitude * rng.choice([-1, 1], size=n_outliers)
            
            # Optional: Add extra randomness to shift magnitude
            shifts += rng.normal(0, shift_magnitude * 0.2, size=n_outliers)
            
            y[outlier_indices] += shifts
    
    return X, y
