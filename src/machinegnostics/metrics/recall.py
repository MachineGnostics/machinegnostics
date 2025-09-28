import numpy as np
import pandas as pd
from machinegnostics.magcal.util.logging import get_logger
import logging

def recall_score(y_true:np.ndarray|pd.Series, 
                 y_pred:np.ndarray|pd.Series, 
                 average='binary', 
                 labels=None,
                 verbose:bool=False) -> float|np.ndarray:
    """
    Computes the recall classification score.

    Recall is the ratio of true positives to the sum of true positives and false negatives.
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
        - None: Return the recall for each class.

    labels : array-like, default=None
        List of labels to include. If None, uses sorted unique labels from y_true and y_pred.

    Returns
    -------
    recall : float or array of floats
        Recall score(s). Float if average is not None, array otherwise.

    Examples
    --------
    >>> y_true = [0, 1, 2, 2, 0]
    >>> y_pred = [0, 0, 2, 2, 0]
    >>> recall_score(y_true, y_pred, average='macro')
    0.8333333333333333

    >>> import pandas as pd
    >>> df = pd.DataFrame({'true': [1, 0, 1], 'pred': [1, 1, 1]})
    >>> recall_score(df['true'], df['pred'], average='binary')
    1.0
    """
    logger = get_logger('recall_score', level=logging.WARNING if not verbose else logging.INFO)
    logger.info("Calculating Recall Score...")
    # If input is a DataFrame, raise error (must select column)
    if isinstance(y_true, pd.DataFrame) or isinstance(y_pred, pd.DataFrame):
        logger.error("y_true and y_pred must be 1D array-like or pandas Series, not DataFrame. Select a column.")
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

    recalls = []
    for label in labels:
        tp = np.sum((y_pred == label) & (y_true == label))
        fn = np.sum((y_pred != label) & (y_true == label))
        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)
        recalls.append(recall)

    recalls = np.array(recalls)
    
    logger.info("Recall Score calculated.")
    if average == 'binary':
        if len(labels) != 2:
            raise ValueError("Binary average is only supported for binary classification with 2 classes.")
        # By convention, use the second label as positive class
        return recalls[1]
    elif average == 'micro':
        tp = sum(np.sum((y_pred == label) & (y_true == label)) for label in labels)
        fn = sum(np.sum((y_pred != label) & (y_true == label)) for label in labels)
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    elif average == 'macro':
        return np.mean(recalls)
    elif average == 'weighted':
        support = np.array([np.sum(y_true == label) for label in labels])
        return np.average(recalls, weights=support)
    elif average is None:
        return recalls
    else:
        raise ValueError(f"Unknown average type: {average}")