import numpy as np
import pandas as pd

def f1_score(y_true:np.ndarray | pd.Series | list,
             y_pred:np.ndarray | pd.Series | list,
             average='binary', 
             labels=None):
    """
    Computes the F1 score for classification tasks.

    The F1 score is the harmonic mean of precision and recall.
    Supports binary and multiclass classification.

    Parameters
    ----------
    y_true : array-like or pandas Series/DataFrame column of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like or pandas Series/DataFrame column of shape (n_samples,)
        Estimated targets as returned by a classifier.

    average : {'binary', 'micro', 'macro', 'weighted', None}, default='binary'
        - 'binary': Only report results for the class specified by `pos_label` (default for binary).
        - 'micro': Calculate metrics globally by counting the total true positives, false negatives and false positives.
        - 'macro': Calculate metrics for each label, and find their unweighted mean.
        - 'weighted': Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).
        - None: Return the F1 score for each class.

    labels : array-like, default=None
        List of labels to include. If None, uses sorted unique labels from y_true and y_pred.

    Returns
    -------
    f1 : float or array of floats
        F1 score(s). Float if average is not None, array otherwise.

    Examples
    --------
    >>> y_true = [0, 1, 2, 2, 0]
    >>> y_pred = [0, 0, 2, 2, 0]
    >>> f1_score(y_true, y_pred, average='macro')
    0.7777777777777777

    >>> import pandas as pd
    >>> df = pd.DataFrame({'true': [1, 0, 1], 'pred': [1, 1, 1]})
    >>> f1_score(df['true'], df['pred'], average='binary')
    0.8
    """
    # If input is a DataFrame, raise error (must select column)
    if isinstance(y_true, pd.DataFrame) or isinstance(y_pred, pd.DataFrame):
        raise ValueError("y_true and y_pred must be 1D array-like or pandas Series, not DataFrame. Select a column.")

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

    # Get unique labels
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.asarray(labels)

    precisions = []
    recalls = []
    for label in labels:
        tp = np.sum((y_pred == label) & (y_true == label))
        fp = np.sum((y_pred == label) & (y_true != label))
        fn = np.sum((y_pred != label) & (y_true == label))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1s = np.where((precisions + recalls) > 0, 2 * precisions * recalls / (precisions + recalls), 0.0)

    if average == 'binary':
        if len(labels) != 2:
            raise ValueError("Binary average is only supported for binary classification with 2 classes.")
        return f1s[1]
    elif average == 'micro':
        tp = sum(np.sum((y_pred == label) & (y_true == label)) for label in labels)
        fp = sum(np.sum((y_pred == label) & (y_true != label)) for label in labels)
        fn = sum(np.sum((y_pred != label) & (y_true == label)) for label in labels)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    elif average == 'macro':
        return np.mean(f1s)
    elif average == 'weighted':
        support = np.array([np.sum(y_true == label) for label in labels])
        return np.average(f1s, weights=support)
    elif average is None:
        return f1s
    else:
        raise ValueError(f"Unknown average type: {average}")