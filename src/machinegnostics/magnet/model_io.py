"""Model saving and loading utilities for Magnet."""

import pickle

def save_weights(model, path):
    """Save model weights to a file."""
    weights = [p.data.copy() for p in model.parameters()]
    with open(path, "wb") as f:
        pickle.dump(weights, f)

def load_weights(model, path):
    """Load model weights from a file."""
    with open(path, "rb") as f:
        weights = pickle.load(f)
    for p, w in zip(model.parameters(), weights):
        p.data[...] = w
