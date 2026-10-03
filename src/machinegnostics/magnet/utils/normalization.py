"""Normalization helpers for MAGNET data pipelines."""

from __future__ import annotations

import numpy as np


def min_max_scale(X, min_val: float = 0.0, max_val: float = 1.0, *, epsilon: float = 1e-8):
    """Scale each feature to a target interval.

    Parameters
    ----------
    X : array-like
        Data to scale.
    min_val, max_val : float, default=(0.0, 1.0)
        Output interval bounds.
    epsilon : float, default=1e-8
        Stabilizer added when a feature range is zero.

    Returns
    -------
    np.ndarray
        Scaled data array.
    """
    if max_val <= min_val:
        raise ValueError("max_val must be greater than min_val")

    X_array = np.asarray(X, dtype=np.float64)
    X_min = np.min(X_array, axis=0)
    X_max = np.max(X_array, axis=0)
    scaled = (X_array - X_min) / (X_max - X_min + epsilon)
    return scaled * (max_val - min_val) + min_val


def standardize(X, *, mean=None, std=None, epsilon: float = 1e-8):
    """Standardize each feature to zero mean and unit variance.

    Parameters
    ----------
    X : array-like
        Data to standardize.
    mean : np.ndarray, optional
        Precomputed feature means.
    std : np.ndarray, optional
        Precomputed feature standard deviations.
    epsilon : float, default=1e-8
        Stabilizer added to very small standard deviations.

    Returns
    -------
    np.ndarray
        Standardized data array.
    """
    X_array = np.asarray(X, dtype=np.float64)
    mean_array = np.asarray(mean, dtype=np.float64) if mean is not None else np.mean(X_array, axis=0)
    std_array = np.asarray(std, dtype=np.float64) if std is not None else np.std(X_array, axis=0)
    safe_std = np.where(std_array < epsilon, std_array + epsilon, std_array)
    return (X_array - mean_array) / safe_std


def l2_normalize(X, axis: int = -1, *, epsilon: float = 1e-8):
    """Normalize vectors to unit L2 norm along a chosen axis."""
    X_array = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X_array, axis=axis, keepdims=True)
    return X_array / np.maximum(norms, epsilon)


__all__ = ["min_max_scale", "standardize", "l2_normalize"]
