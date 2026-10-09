import numpy as np

def accuracy(y_pred, y_true):
    """y_pred: probabilities/logits (N, C). y_true: one-hot (N, C) or int labels (N,)."""
    pred_labels = np.argmax(y_pred, axis=-1)
    true_labels = np.argmax(y_true, axis=-1) if y_true.ndim > 1 else y_true
    return np.mean(pred_labels == true_labels)