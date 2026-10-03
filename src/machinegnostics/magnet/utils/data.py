"""Data pipeline helpers for MAGNET models."""

from __future__ import annotations

import numpy as np

from .misc import validate_same_length


def shuffle_data(X, y):
    """Shuffle features and targets with a shared random permutation.

    Parameters
    ----------
    X, y : array-like
        Input features and targets with matching sample counts.

    Returns
    -------
    tuple of np.ndarray
        Shuffled ``(X, y)`` arrays.
    """
    X_array = np.asarray(X)
    y_array = np.asarray(y)
    validate_same_length(X_array, y_array)
    indices = np.random.permutation(len(X_array))
    return X_array[indices], y_array[indices]


def create_batches(X, y, batch_size: int, *, shuffle: bool = True):
    """Yield mini-batches for iterative training.

    Parameters
    ----------
    X, y : array-like
        Input features and targets with matching sample counts.
    batch_size : int
        Number of samples per batch.
    shuffle : bool, default=True
        Whether to randomize sample order before batching.

    Yields
    ------
    tuple of np.ndarray
        ``(X_batch, y_batch)`` pairs.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    X_array = np.asarray(X)
    y_array = np.asarray(y)
    validate_same_length(X_array, y_array)

    if shuffle:
        X_array, y_array = shuffle_data(X_array, y_array)

    for start in range(0, len(X_array), batch_size):
        stop = start + batch_size
        yield X_array[start:stop], y_array[start:stop]


def normalize(X, mean=None, std=None, *, epsilon: float = 1e-8):
    """Normalize features using z-score scaling.

    Parameters
    ----------
    X : array-like
        Feature matrix or vector to normalize.
    mean : np.ndarray, optional
        Precomputed feature means. If omitted, computed from ``X``.
    std : np.ndarray, optional
        Precomputed feature standard deviations. If omitted, computed from
        ``X``.
    epsilon : float, default=1e-8
        Small positive value added to the denominator for numerical stability.

    Returns
    -------
    tuple
        ``(X_normalized, mean, std)``.
    """
    X_array = np.asarray(X, dtype=np.float64)
    mean_array = np.asarray(mean, dtype=np.float64) if mean is not None else np.mean(X_array, axis=0)
    std_array = np.asarray(std, dtype=np.float64) if std is not None else np.std(X_array, axis=0)
    safe_std = np.where(std_array < epsilon, std_array + epsilon, std_array)
    normalized = (X_array - mean_array) / safe_std
    return normalized, mean_array, safe_std


def split_train_test(X, y, test_size: float | int = 0.2, *, shuffle: bool = True):
    """Split data into train and test partitions.

    Parameters
    ----------
    X, y : array-like
        Features and targets with matching sample counts.
    test_size : float or int, default=0.2
        Fraction of samples to reserve for testing, or an explicit number of
        test samples.
    shuffle : bool, default=True
        Whether to shuffle before splitting.

    Returns
    -------
    tuple
        ``(X_train, y_train, X_test, y_test)``.
    """
    X_array = np.asarray(X)
    y_array = np.asarray(y)
    validate_same_length(X_array, y_array)

    n_samples = len(X_array)
    if isinstance(test_size, float):
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size as a float must be between 0 and 1")
        n_test = int(round(n_samples * test_size))
    else:
        n_test = int(test_size)

    if n_test <= 0 or n_test >= n_samples:
        raise ValueError("test_size must leave at least one sample in each split")

    indices = np.arange(n_samples)
    if shuffle:
        indices = np.random.permutation(n_samples)

    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X_array[train_idx], y_array[train_idx], X_array[test_idx], y_array[test_idx]


__all__ = ["shuffle_data", "create_batches", "normalize", "split_train_test"]
