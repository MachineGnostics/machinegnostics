# confusion_matrix: Confusion Matrix Metric

The `confusion_matrix` function computes the confusion matrix for classification tasks. This metric provides a summary of prediction results, showing how many samples were correctly or incorrectly classified for each class.

---

## Overview

A confusion matrix is a table that is often used to describe the performance of a classification model.  
- Each row represents the actual class.
- Each column represents the predicted class.
- Entry (i, j) is the number of samples with true label i and predicted label j.

This metric helps you understand the types of errors your classifier is making and is essential for evaluating classification accuracy, precision, recall, and other related metrics.

- Compatibility: Accepts NumPy arrays and Pandas Series for `y_true` and `y_pred`.

---

## Parameters

| Parameter   | Type         | Description                                                                                 | Default      |
|-------------|--------------|---------------------------------------------------------------------------------------------|--------------|
| `y_true`    | array-like or pandas Series | Ground truth (correct) target values. Shape: (n_samples,)                        | Required     |
| `y_pred`    | array-like or pandas Series | Estimated targets as returned by a classifier. Shape: (n_samples,)                | Required     |
| `labels`    | array-like   | List of labels to index the matrix. If None, uses all labels in sorted order.               | `None`       |
| `verbose`   | bool         | If True, enables detailed logging for debugging.                                            | `False`      |

---

## Returns

- **ndarray**  
  Confusion matrix of shape (n_classes, n_classes). Entry (i, j) is the number of samples with true label i and predicted label j.

---

## Raises

- **ValueError**  
  If input arrays are not the same shape, are empty, are not 1D, or contain NaN/Inf.

---

## Example Usage

```python
import numpy as np
from machinegnostics.metrics import confusion_matrix

# Example 1: Basic usage
y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
cm = confusion_matrix(y_true, y_pred)
print(cm)
# Output:
# array([[2, 0, 0],
#        [0, 0, 1],
#        [1, 0, 2]])

# Example 2: With custom labels
cm_custom = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
print(cm_custom)
```

---

## Notes

- Compatibility: Accepts NumPy arrays, lists, or Pandas Series.
- Both `y_true` and `y_pred` must be 1D, have the same shape, and must not be empty or contain NaN/Inf.
- If `labels` is not provided, all unique labels in `y_true` and `y_pred` are used in sorted order.
- The confusion matrix is essential for computing other metrics such as accuracy, precision, recall, and F1-score.

---

**Author:** Nirmal Parmar  
**Date:** 2025-09-24

---
