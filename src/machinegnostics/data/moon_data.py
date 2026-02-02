import numpy as np

def make_moons_check_data(n_samples=30, noise=None, seed=42):
    """
    Generates synthetic 'two moons' classification data for validating Machine Gnostics models.
    
    This function creates two interleaving half circles, a classic dataset for 
    visualizing clustering and classification algorithms, especially for non-linear 
    decision boundaries.

    Parameters
    ----------
    n_samples : int, optional
        The total number of data points to generate. Default is 30.
    noise : float or None, optional
        Standard deviation of Gaussian noise added to the data. 
        None implies no noise. Default is None.
    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    X : numpy.ndarray
        The input feature array of shape (n_samples, 2).
    y : numpy.ndarray
        The target label array of shape (n_samples,).
        0 for the upper moon, 1 for the lower moon.

    Example
    -------
    >>> from machinegnostics.data.moon_data import make_moons_check_data
    >>> X, y = make_moons_check_data(n_samples=100, noise=0.1)
    >>> print(f"X shape: {X.shape}, Unique classes: {np.unique(y)}")
    X shape: (100, 2), Unique classes: [0 1]
    """
    rng = np.random.default_rng(seed)

    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    # Upper moon (Class 0)
    # theta in [0, pi]
    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    
    # Lower moon (Class 1)
    # theta in [0, pi]
    # Logic: 1 - cos(theta) shifts x from [-1, 1] to [0, 2]? No.
    # cos goes 1 -> -1. 1 - cos goes 0 -> 2.
    # 1 - sin(theta) - 0.5: sin goes 0 -> 1 -> 0. 1-sin goes 1 -> 0 -> 1. 
    # -0.5 shift: 0.5 -> -0.5 -> 0.5.
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 0.5

    X = np.vstack([
        np.column_stack([outer_circ_x, outer_circ_y]),
        np.column_stack([inner_circ_x, inner_circ_y])
    ])
    
    y = np.hstack([
        np.zeros(n_samples_out, dtype=int),
        np.ones(n_samples_in, dtype=int)
    ])

    if noise is not None:
        X += rng.normal(scale=noise, size=X.shape)

    # Shuffle
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    return X[indices], y[indices]
