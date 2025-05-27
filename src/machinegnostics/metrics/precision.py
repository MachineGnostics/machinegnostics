import numpy as np
import pandas as pd

def precision_score(y_true, y_pred, average='binary', labels=None):
    """
    Computes the precision classification score.

    Precision is the ratio of true positives to the sum of true and false positives.
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
        - None: Return the precision for each class.

    labels : array-like, default=None
        List of labels to include. If None, uses sorted unique labels from y_true and y_pred.

    Returns
    -------
    precision : float or array of floats
        Precision score(s). Float if average is not None, array otherwise.

    Examples
    --------
    >>> y_true = [0, 1, 2, 2, 0]
    >>> y_pred = [0, 0, 2, 2, 0]
    >>> precision_score(y_true, y_pred, average='macro')
    0.8333333333333333

    >>> import pandas as pd
    >>> df = pd.DataFrame({'true': [1, 0, 1], 'pred': [1, 1, 1]})
    >>> precision_score(df['true'], df['pred'], average='binary')
    0.6666666666666666
    """
    # If input is a DataFrame, raise error (must select column)
    if isinstance(y_true, pd.DataFrame) or isinstance(y_pred, pd.DataFrame):
        raise ValueError("y_true and y_pred must be 1D array-like or pandas Series, not DataFrame. Select a column.")

    # Convert pandas Series to numpy array
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("Shape of y_true and y_pred must be the same.")

    # Get unique labels
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.asarray(labels)

    precisions = []
    for label in labels:
        tp = np.sum((y_pred == label) & (y_true == label))
        fp = np.sum((y_pred == label) & (y_true != label))
        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)
        precisions.append(precision)

    precisions = np.array(precisions)

    if average == 'binary':
        if len(labels) != 2:
            raise ValueError("Binary average is only supported for binary classification with 2 classes.")
        # By convention, use the second label as positive class
        return precisions[1]
    elif average == 'micro':
        tp = sum(np.sum((y_pred == label) & (y_true == label)) for label in labels)
        fp = sum(np.sum((y_pred == label) & (y_true != label)) for label in labels)
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    elif average == 'macro':
        return np.mean(precisions)
    elif average == 'weighted':
        support = np.array([np.sum(y_true == label) for label in labels])
        return np.average(precisions, weights=support)
    elif average is None:
        return precisions
    else:
        raise ValueError(f"Unknown average type: {average}")