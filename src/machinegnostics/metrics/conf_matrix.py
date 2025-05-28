import numpy as np
import pandas as pd

def confusion_matrix(y_true:np.ndarray | pd.Series,
                     y_pred:np.ndarray | pd.Series, 
                     labels=None):
    """
    Computes the confusion matrix to evaluate the accuracy of a classification.

    By definition, entry (i, j) in the confusion matrix is the number of observations
    actually in class i but predicted to be in class j.

    Parameters
    ----------
    y_true : array-like or pandas Series/DataFrame column of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like or pandas Series/DataFrame column of shape (n_samples,)
        Estimated targets as returned by a classifier.

    labels : array-like, default=None
        List of labels to index the matrix. This may be used to reorder or select a subset of labels.
        If None, labels that appear at least once in y_true or y_pred are used in sorted order.

    Returns
    -------
    cm : ndarray of shape (n_classes, n_classes)
        Confusion matrix whose i-th row and j-th column entry indicates the number of samples with
        true label being i-th class and predicted label being j-th class.

    Examples
    --------
    >>> y_true = [2, 0, 2, 2, 0, 1]
    >>> y_pred = [0, 0, 2, 2, 0, 2]
    >>> confusion_matrix(y_true, y_pred)
    array([[2, 0, 0],
           [0, 0, 1],
           [1, 0, 2]])
    """

    # Convert pandas Series to numpy array
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    # Convert to numpy arrays and flatten
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if y_true.shape != y_pred.shape:
        raise ValueError("Shape of y_true and y_pred must be the same.")

    # Determine labels
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.asarray(labels)
    n_labels = len(labels)
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    # Initialize confusion matrix
    cm = np.zeros((n_labels, n_labels), dtype=int)

    # Populate confusion matrix
    for true, pred in zip(y_true, y_pred):
        if true in label_to_index and pred in label_to_index:
            i = label_to_index[true]
            j = label_to_index[pred]
            cm[i, j] += 1

    return cm