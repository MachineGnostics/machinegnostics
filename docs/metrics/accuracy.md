# accuracy_score: Classification Accuracy Metric

The `accuracy_score` function computes the accuracy of classification models by comparing predicted labels to true labels. It is a fundamental metric for evaluating the performance of classifiers in binary and multiclass settings.

---

## Overview

Accuracy is defined as the proportion of correct predictions among the total number of cases examined. It is a simple yet powerful metric for assessing how well a model is performing, especially when the classes are balanced.

- Compatibility: Accepts NumPy arrays and Pandas Series for `y_true` and `y_pred`.

---

## Parameters

| Parameter   | Type                        | Description                                                              |
| ----------- | --------------------------- | ------------------------------------------------------------------------ |
| `y_true`  | array-like or pandas Series | Ground truth (correct) target values. Shape: (n_samples,)                |
| `y_pred`  | array-like or pandas Series | Estimated target values as returned by a classifier. Shape: (n_samples,) |
| `verbose` | bool                        | Print detailed progress, warnings, and results                           |

- Both `y_true` and `y_pred` can be numpy arrays, lists, or pandas Series.
- If a pandas DataFrame is passed, a `ValueError` is raised (select a column instead).

---

## Returns

- **accuracy**: `float`
  The accuracy score as a float in the range [0, 1].

---

## Raises

- **ValueError**
  - If `y_true` or `y_pred` is a pandas DataFrame (must select a column).
  - If the shapes of `y_true` and `y_pred` do not match.

---

## Example Usage

```python
from machinegnostics.metrics import accuracy_score

# Example 1: Using lists
y_true = [0, 1, 2, 2, 0]
y_pred = [0, 0, 2, 2, 0]
print(accuracy_score(y_true, y_pred))  # Output: 0.8

# Example 2: Using pandas Series
import pandas as pd
df = pd.DataFrame({'true': [1, 0, 1], 'pred': [1, 1, 1]})
print(accuracy_score(df['true'], df['pred']))
```

---

## Notes

- Compatibility: Accepts NumPy arrays, lists, or Pandas Series.
- If you pass a pandas DataFrame, you must select a column (e.g., `df['col']`), not the whole DataFrame.
- The accuracy metric is most informative when the dataset is balanced. For imbalanced datasets, consider additional metrics such as precision, recall, or F1 score.

---