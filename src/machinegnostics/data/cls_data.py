import numpy as np

def make_classification_check_data(n_samples=30, n_features=2, n_classes=2, separability=2.0, seed=42):
    """
    Generates synthetic classification data for validating Machine Gnostics models.

    This function creates a simple blob-based classification dataset using Gaussian 
    distributions. It is designed to be a 'hello world' test for checking if a 
    classification model is functioning correctly.

    Parameters
    ----------
    n_samples : int, optional
        The total number of data points to generate. Default is 30.
    n_features : int, optional
        The number of input features (dimensions) for each sample. Default is 2.
    n_classes : int, optional
        The number of distinct classes (labels). Default is 2.
    separability : float, optional
        Controls the distance between the centers of the class blobs. 
        Higher values make the classification task easier. Default is 2.0.
    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    X : numpy.ndarray
        The input feature array of shape (n_samples, n_features).
    y : numpy.ndarray
        The target label array of shape (n_samples,).

    Example
    -------
    >>> from machinegnostics.data.cls_data import make_classification_check_data
    >>> X, y = make_classification_check_data(n_samples=50, n_classes=3)
    >>> print(f"X shape: {X.shape}, Unique classes: {np.unique(y)}")
    X shape: (50, 2), Unique classes: [0 1 2]
    """
    rng = np.random.default_rng(seed)
    
    samples_per_class = n_samples // n_classes
    X_list = []
    y_list = []
    
    # Generate random centers for each class
    # We multiply by separability to space them out
    centers = rng.standard_normal((n_classes, n_features)) * separability
    
    for class_idx in range(n_classes):
        # Create samples around the center with some variance
        # Handle the remainder samples for the last class to ensure total match n_samples
        if class_idx == n_classes - 1:
            current_n = n_samples - len(X_list) * samples_per_class
        else:
            current_n = samples_per_class
            
        noise = rng.standard_normal((current_n, n_features))
        # Reduce spread of blobs (0.7) to make them tighter than the separation
        samples = centers[class_idx] + noise * 0.7
        
        X_list.append(samples)
        y_list.append(np.full(current_n, class_idx))
        
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    # Shuffle the data so classes aren't ordered
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    
    return X[indices], y[indices]
