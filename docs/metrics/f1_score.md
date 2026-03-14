# f1_score: Classification F1 Score Metric

The `f1_score` function computes the F1 score for classification models, supporting both binary and multiclass settings. The F1 score is the harmonic mean of precision and recall, providing a balanced measure that is especially useful when classes are imbalanced.

---

## Overview

The F1 score combines precision and recall into a single metric by taking their harmonic mean.

This metric is particularly important when you want to balance the trade-off between precision and recall, such as in information retrieval, medical diagnosis, and fraud detection.

- Compatibility: Accepts NumPy arrays and Pandas Series for `y_true` and `y_pred`.

---

## Parameters

| Parameter   | Type                                           | Default  | Description                                                                           |
| ----------- | ---------------------------------------------- | -------- | ------------------------------------------------------------------------------------- |
| `y_true`  | array-like or pandas Series                    | —       | Ground truth (correct) target values. Shape: (n_samples,)                             |
| `y_pred`  | array-like or pandas Series                    | —       | Estimated target values as returned by a classifier. Shape: (n_samples,)              |
| `average` | {'binary', 'micro', 'macro', 'weighted', None} | 'binary' | Determines the type of averaging performed on the data. See below for details.        |
| `labels`  | array-like or None                             | None     | List of labels to include. If None, uses sorted unique labels from y_true and y_pred. |
| `verbose` | bool                                           | True     | Print detailed progress, warnings, and results                                        |

### Averaging Options

- **'binary'**: Only report results for the positive class (default for binary classification).
- **'micro'**: Calculate metrics globally by counting the total true positives, false negatives, and false positives.
- **'macro'**: Calculate metrics for each label, and find their unweighted mean.
- **'weighted'**: Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).
- **None**: Return the F1 score for each class as an array.

---

## Returns

- **f1**: `float` or `array of floats`
  F1 score(s). Returns a float if `average` is not None, otherwise returns an array of F1 values for each class.

---

## Raises

- **ValueError**
  - If `y_true` or `y_pred` is a pandas DataFrame (must select a column).
  - If the shapes of `y_true` and `y_pred` do not match.
  - If `average='binary'` but the problem is not binary classification.
  - If `average` is not a recognized option.

---

## Example Usage

```python
from machinegnostics.metrics import f1_score

# Example 1: Macro-averaged F1 for multiclass
y_true = [0, 1, 2, 2, 0]
y_pred = [0, 0, 2, 2, 0]
print(f1_score(y_true, y_pred, average='macro'))

# Example 2: Binary F1 with pandas Series
import pandas as pd
df = pd.DataFrame({'true': [1, 0, 1], 'pred': [1, 1, 1]})
print(f1_score(df['true'], df['pred'], average='binary'))
```

---

## Notes

- Compatibility: Accepts NumPy arrays, lists, or Pandas Series.
- If you pass a pandas DataFrame, you must select a column (e.g., `df['col']`), not the whole DataFrame.
- For binary classification, by convention, the second label is treated as the positive class.
- For imbalanced datasets, consider using `average='weighted'` to account for class support.

---
