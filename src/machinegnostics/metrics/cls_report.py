import numpy as np
import pandas as pd
from machinegnostics.metrics import precision_score, recall_score, f1_score
from machinegnostics.magcal.util.logging import get_logger
import logging

def classification_report(
    y_true:np.ndarray | pd.Series,
    y_pred:np.ndarray | pd.Series,
    labels=None,
    target_names=None,
    digits=2,
    output_dict=False,
    verbose: bool = False
):
    """
    Builds a text summary or dictionary of the precision, recall, F1 score, and support for each class.

    Uses the precision_score, recall_score, and f1_score functions for consistency.

    Parameters
    ----------
    y_true : array-like or pandas Series/DataFrame column of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like or pandas Series/DataFrame column of shape (n_samples,)
        Estimated targets as returned by a classifier.

    labels : array-like, default=None
        List of labels to include in the report. If None, uses sorted unique labels from y_true and y_pred.

    target_names : list of str, default=None
        Optional display names matching the labels (same order).

    digits : int, default=2
        Number of digits for formatting output.

    output_dict : bool, default=False
        If True, return output as a dict for programmatic use. If False, return as a formatted string.

    verbose : bool, optional
        If True, enables detailed logging for debugging purposes. Default is False.

    Returns
    -------
    report : str or dict
        Text summary or dictionary of the precision, recall, F1 score for each class.
    """
    logger = get_logger('classification_report', level=logging.WARNING if not verbose else logging.INFO)
    logger.info("Generating Classification Report...")
    # Convert pandas Series to numpy array
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    # Convert to numpy arrays and flatten
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if y_true.shape != y_pred.shape:
        logger.error("Shape of y_true and y_pred must be the same.")
        raise ValueError("Shape of y_true and y_pred must be the same.")

    # Get unique labels
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.asarray(labels)

    n_labels = len(labels)
    if target_names is not None:
        if len(target_names) != n_labels:
            logger.error("target_names length must match number of labels")
            raise ValueError("target_names length must match number of labels")
    else:
        target_names = [str(label) for label in labels]

    # Use your metric functions for each class
    precisions = precision_score(y_true, y_pred, average=None, labels=labels)
    recalls = recall_score(y_true, y_pred, average=None, labels=labels)
    f1s = f1_score(y_true, y_pred, average=None, labels=labels)
    supports = np.array([(y_true == label).sum() for label in labels])

    # Weighted averages
    total_support = supports.sum()
    avg_precision = np.average(precisions, weights=supports) if total_support > 0 else 0.0
    avg_recall = np.average(recalls, weights=supports) if total_support > 0 else 0.0
    avg_f1 = np.average(f1s, weights=supports) if total_support > 0 else 0.0

    if output_dict:
        report = {}
        for i, label in enumerate(labels):
            report[target_names[i]] = {
                "precision": round(precisions[i], digits),
                "recall": round(recalls[i], digits),
                "f1-score": round(f1s[i], digits),
                "support": int(supports[i])
            }
        report["avg/total"] = {
            "precision": round(avg_precision, digits),
            "recall": round(avg_recall, digits),
            "f1-score": round(avg_f1, digits),
            "support": int(total_support)
        }
        return report

    # Build report string
    header = f"{'Class':<15}{'Precision':>10}{'Recall':>10}{'F1-score':>10}{'Support':>10}\n"
    report = header
    report += "=" * len(header) + "\n"
    for i in range(n_labels):
        report += (
            f"{target_names[i]:<15}"
            f"{precisions[i]:>10.{digits}f}"
            f"{recalls[i]:>10.{digits}f}"
            f"{f1s[i]:>10.{digits}f}"
            f"{supports[i]:>10}\n"
        )
    report += "=" * len(header) + "\n"
    report += (
        f"{'Avg/Total':<15}"
        f"{avg_precision:>10.{digits}f}"
        f"{avg_recall:>10.{digits}f}"
        f"{avg_f1:>10.{digits}f}"
        f"{total_support:>10}\n"
    )

    logger.info("Classification Report generated.")
    return report