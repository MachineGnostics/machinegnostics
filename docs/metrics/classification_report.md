# classification_report: Classification Metrics Summary

The `classification_report` function generates a comprehensive summary of key classification metrics—precision, recall, F1 score, and support—for each class in your dataset. It supports both string and dictionary output formats, making it suitable for both human-readable reports and programmatic analysis.

---

## Overview

This function provides a detailed breakdown of classifier performance for each class, including:

- **Precision**: Proportion of positive identifications that were actually correct.
- **Recall**: Proportion of actual positives that were correctly identified.
- **F1 Score**: Harmonic mean of precision and recall.
- **Support**: Number of true instances for each class.

It also computes weighted averages across all classes.

- Compatibility: Accepts NumPy arrays and Pandas Series for `y_true` and `y_pred`.

---

## Parameters

| Parameter         | Type                        | Default | Description                                                                                         |
| ----------------- | --------------------------- | ------- | --------------------------------------------------------------------------------------------------- |
| `y_true`        | array-like or pandas Series | —      | Ground truth (correct) target values. Shape: (n_samples,)                                           |
| `y_pred`        | array-like or pandas Series | —      | Estimated target values as returned by a classifier. Shape: (n_samples,)                            |
| `labels`        | array-like or None          | None    | List of labels to include in the report. If None, uses sorted unique labels from y_true and y_pred. |
| `target_names`  | list of str or None         | None    | Optional display names matching the labels (same order).                                            |
| `digits`        | int                         | 2       | Number of digits for formatting output.                                                             |
| `output_dict`   | bool                        | False   | If True, return output as a dict. If False, return as a formatted string.                           |
| `zero_division` | {0, 1, 'warn'}              | 0       | Value to return when there is a zero division (no predicted samples for a class).                   |
| `verbose`           | bool                  | True      | Print detailed progress, warnings, and results                   |

---

## Returns

- **report**: `str` or `dict`
  Text summary or dictionary of the precision, recall, F1 score, and support for each class.

---

## Example Usage

```python
from machinegnostics.metrics import classification_report

y_true = [0, 1, 2, 2, 0]
y_pred = [0, 0, 2, 2, 0]

# String report
print(classification_report(y_true, y_pred))

# Dictionary report
report_dict = classification_report(y_true, y_pred, output_dict=True)
print(report_dict)
```

---

## Output Example

**String Output:**

```
Class             Precision    Recall   F1-score    Support
==========================================================
0                    1.00      0.50      0.67          2
1                    0.00      0.00      0.00          1
2                    1.00      1.00      1.00          2
==========================================================
Avg/Total            0.80      0.60      0.67          5
```

**Dictionary Output:**

```python
{
  '0': {'precision': 1.0, 'recall': 0.5, 'f1-score': 0.67, 'support': 2},
  '1': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 1},
  '2': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 2},
  'avg/total': {'precision': 0.8, 'recall': 0.6, 'f1-score': 0.67, 'support': 5}
}
```

---

## Notes

- The function uses `precision_score`, `recall_score`, and `f1_score` from the Machine Gnostics metrics module for consistency.
- If `target_names` is provided, its length must match the number of labels.
- For imbalanced datasets, the weighted average provides a more informative summary than the unweighted mean.
- The `zero_division` parameter controls the behavior when a class has no predicted samples.

---

