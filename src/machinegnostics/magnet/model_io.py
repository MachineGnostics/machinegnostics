"""Model saving and loading utilities for Magnet."""

import pickle

def save_weights(model, path):
    """
    Save model weights to a file.

    Parameters
    ----------
    model : object
        Model instance with a `parameters()` method (e.g., Sequential).
    path : str
        Path to the file where weights will be saved (e.g., 'model_weights.pkl').

    Example
    -------
    >>> from machinegnostics.magnet.model_io import save_weights
    >>> save_weights(model, 'my_weights.pkl')

    Notes
    -----
    Only the model's parameters (weights) are saved. To restore weights, use `load_weights(model, path)` on a model with the same architecture.
    """
    weights = [p.data.copy() for p in model.parameters()]
    with open(path, "wb") as f:
        pickle.dump(weights, f)

def load_weights(model, path):
    """
    Load model weights from a file.

    Parameters
    ----------
    model : object
        Model instance with a `parameters()` method (e.g., Sequential).
    path : str
        Path to the file from which weights will be loaded (e.g., 'model_weights.pkl').

    Example
    -------
    >>> from machinegnostics.magnet.model_io import load_weights
    >>> load_weights(model, 'my_weights.pkl')

    Notes
    -----
    The model architecture must match the saved weights. This function loads weights into the provided model instance.
    """
    with open(path, "rb") as f:
        weights = pickle.load(f)
    for p, w in zip(model.parameters(), weights):
        p.data[...] = w
