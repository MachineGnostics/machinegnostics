'''
Silhouette Score - Machine Gnostics Framework

Machine Gnostics

Author: Nirmal Parmar
'''

import logging
from machinegnostics.magcal.util.logging import get_logger
import numpy as np
from typing import Union, List
from machinegnostics.magcal.util.narwhals_df import narwhalify

@narwhalify
def silhouette_score(X: Union[np.ndarray, List], 
                     labels: Union[np.ndarray, List],
                     verbose: bool = False) -> float:
    """
    Calculate the mean Silhouette Coefficient for all samples.
    
    The Silhouette Coefficient is calculated using the mean intra-cluster
    distance (a) and the mean nearest-cluster distance (b) for each sample.
    The Silhouette Coefficient for a sample is (b - a) / max(a, b).
    
    Parameters:
    -----------
    X : array-like or dataframe of shape (n_samples, n_features)
        Feature array; accepts NumPy arrays or dataframe types.
    labels : array-like or series of shape (n_samples,)
        Cluster labels for each sample; accepts arrays or series types.
    verbose : bool, optional
        If True, enables detailed logging for debugging purposes.
        
    Returns:
    --------
    float
        Mean Silhouette Coefficient.
        
    Raises:
    -------
    TypeError
        If inputs are not array-like.
    ValueError
        If input dimensions are incorrect or contain invalid values.
        
    Example:
    --------
    >>> from machinegnostics.metrics import silhouette_score
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> labels = np.array([0, 0, 0, 1, 1, 1])
    >>> silhouette_score(X, labels)
    """
    logger = get_logger('silhouette_score', level=logging.WARNING if not verbose else logging.INFO)
    logger.info("Calculating silhouette score...")

    # Validate inputs
    if not isinstance(X, (list, tuple, np.ndarray)):
        logger.error("X must be array-like (list, tuple, or numpy array).")
        raise TypeError("X must be array-like (list, tuple, or numpy array).")
    if not isinstance(labels, (list, tuple, np.ndarray)):
        logger.error("labels must be array-like (list, tuple, or numpy array).")
        raise TypeError("labels must be array-like (list, tuple, or numpy array).")

    # Convert to numpy arrays
    X = np.asarray(X)
    labels = np.asarray(labels).flatten()
    n_samples = len(X)
    
    # Check dimensions
    if X.ndim != 2:
        logger.error("X must be a 2D array of shape (n_samples, n_features).")
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
    if len(labels) != n_samples:
        logger.error("Number of labels must match number of samples in X.")
        raise ValueError(f"Number of labels ({len(labels)}) must match number of samples in X ({n_samples}).")
    
    # Check for empty or invalid data
    if n_samples == 0:
        logger.error("Input data X is empty.")
        raise ValueError("Input data X is empty.")
    if np.any(np.isnan(X)) or np.any(np.isnan(labels)):
        logger.error("Input data contains NaN values.")
        raise ValueError("Input data contains NaN values.")
    if np.any(np.isinf(X)):
        logger.error("Input data X contains Inf values.")
        raise ValueError("Input data X contains Inf values.")

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        logger.warning("Number of unique labels is less than 2. Returning 0.0.")
        return 0.0
    if len(unique_labels) == n_samples:
         logger.warning("Number of unique labels equals number of samples. Returning 0.0.")
         return 0.0

    logger.info(f"Computing pairwise distances for {n_samples} samples...")
    # Calculate pairwise Euclidean distances using pure NumPy broadcasting
    # dist[i, j] = sqrt((x[i] - x[j])^2)
    # Using the identity: ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
    # This is faster than a double loop but can be memory intensive for very large N.
    # For standard usage, this is vector-efficient.
    X_sq = np.sum(X**2, axis=1, keepdims=True)
    distances_sq = X_sq + X_sq.T - 2 * np.dot(X, X.T)
    # Numerical stability: essentially zero values might become slightly negative
    distances_sq = np.maximum(distances_sq, 0)
    distances = np.sqrt(distances_sq)
    
    silhouette_scores = np.zeros(n_samples)
    
    logger.info("Computing silhouette scores per sample...")
    for i in range(n_samples):
        # --- Calculate a(i) ---
        # Mean distance to other points in the same cluster
        same_cluster_mask = (labels == labels[i])
        # We need to exclude the point itself from the mean calculation
        # The distance to itself is 0, so sum/count-1 works, 
        # or we just mask it out.
        same_cluster_mask[i] = False 
        
        cluster_size = np.sum(same_cluster_mask)
        
        if cluster_size == 0:
            # Cluster has only one element (this one)
            silhouette_scores[i] = 0.0
            continue
            
        a_i = np.mean(distances[i, same_cluster_mask])
        
        # --- Calculate b(i) ---
        # Mean distance to points in the nearest other cluster
        b_i = np.inf
        
        for label in unique_labels:
            if label == labels[i]:
                continue
            
            other_cluster_mask = (labels == label)
            if np.any(other_cluster_mask):
                mean_dist_cluster = np.mean(distances[i, other_cluster_mask])
                if mean_dist_cluster < b_i:
                    b_i = mean_dist_cluster
                    
        # --- Calculate score ---
        if b_i == np.inf: 
            # Should not happen given check for len(unique_labels) < 2
            silhouette_scores[i] = 0.0
        else:
            silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i)
             
    final_score = np.mean(silhouette_scores)
    logger.info(f"Mean Silhouette Score calculated: {final_score}")
    
    return float(final_score)