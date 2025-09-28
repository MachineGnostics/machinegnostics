import numpy as np
import logging
from machinegnostics.magcal.util.logging import get_logger

def train_test_split(X:np.ndarray, y=None, test_size=0.25, shuffle=True, random_seed=None, verbose: bool = False):
    """
    Splits arrays or matrices into random train and test subsets.

    Parameters
    ----------
    X : array-like (list, tuple, or np.ndarray)
        Feature data to be split. Must be indexable and of consistent length.
    
    y : array-like or None, optional (default=None)
        Target data to be split alongside X. Must be same length as X.

    test_size : float or int, optional (default=0.25)
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
        If int, represents the absolute number of test samples.

    shuffle : bool, optional (default=True)
        Whether or not to shuffle the data before splitting.

    random_seed : int or None, optional (default=None)
        Controls the shuffling applied to the data before splitting.

    verbose : bool, optional (default=False)
        If True, enables detailed logging.

    Returns
    -------
    X_train, X_test : np.ndarray
        Train-test split of X.
    
    y_train, y_test : np.ndarray or None
        Train-test split of y. If y is None, these will also be None.

    Raises
    ------
    ValueError
        If inputs are invalid or test_size is not appropriate.

    Example
    -------
    >>> import numpy as np
    >>> from machinegnostics.models import train_test_split
    >>> X = np.arange(20).reshape(10, 2)
    >>> y = np.arange(10)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_seed=42)
    >>> print("X_train:", X_train)
    >>> print("X_test:", X_test)
    >>> print("y_train:", y_train)
    >>> print("y_test:", y_test)
    """
    logger = get_logger('train_test_split', level=logging.WARNING if not verbose else logging.INFO)
    logger.info("Starting train_test_split...")

    # Convert inputs to np arrays
    X = np.asarray(X)
    if y is not None:
        y = np.asarray(y)

    # Validate shapes
    if y is not None and len(X) != len(y):
        logger.error(f"X and y must have the same number of samples, got {len(X)} and {len(y)}.")
        raise ValueError(f"X and y must have the same number of samples, got {len(X)} and {len(y)}.")

    n_samples = len(X)

    # Validate and compute test size
    if isinstance(test_size, float):
        if not 0.0 < test_size < 1.0:
            logger.error("If test_size is a float, it must be between 0.0 and 1.0.")
            raise ValueError("If test_size is a float, it must be between 0.0 and 1.0.")
        n_test = int(np.ceil(test_size * n_samples))
    elif isinstance(test_size, int):
        if not 0 < test_size < n_samples:
            logger.error("If test_size is an int, it must be between 1 and len(X) - 1.")
            raise ValueError("If test_size is an int, it must be between 1 and len(X) - 1.")
        n_test = test_size
    else:   
        logger.error("test_size must be either a float or an int.")
        raise TypeError("test_size must be either a float or an int.")

    n_train = n_samples - n_test

    # Create indices and shuffle
    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(seed=random_seed)
        rng.shuffle(indices)

    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train = X[train_idx]
    X_test = X[test_idx]

    if y is not None:
        y_train = y[train_idx]
        y_test = y[test_idx]
    else:
        y_train = y_test = None
    logger.info("Completed train_test_split.")

    return X_train, X_test, y_train, y_test
