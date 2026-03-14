# root_mean_squared_error: Root Mean Squared Error (RMSE) Metric

The `root_mean_squared_error` function computes the Root Mean Squared Error (RMSE) between true and predicted values. RMSE is a widely used regression metric that measures the square root of the average of the squared differences between predicted and actual values.

> Unlike traditional error metrics that use the statistical mean, Machine Gnostics metrics are computed using the gnostic mean. The gnostic mean is a robust, assumption-free measure designed to provide deeper insight and reliability, especially in the presence of outliers or non-normal data.
> This approach ensures that error metrics reflect the true structure and diagnostic properties of your data, in line with the principles of Mathematical Gnostics.

---

## Overview

Root Mean Squared Error is defined as the square root of the mean squared error.

RMSE provides an interpretable measure of prediction error in the same units as the target variable. Lower RMSE values indicate better model performance.

---

## Parameters

| Parameter   | Type                           | Description                                    |
| ----------- | ------------------------------ | ---------------------------------------------- |
| `y_true`  | array-like or Pandas Series     | True values (targets).                         |
| `y_pred`  | array-like or Pandas Series     | Predicted values.                              |
| `verbose` | bool                            | Print detailed progress, warnings, and results |

---

## Returns

- **float**
  The square root of the average of squared errors between actual and predicted values.

---

## Raises

- **TypeError** If `y_true` or `y_pred` are not array-like (NumPy arrays, lists) or Pandas Series.
- **ValueError** If inputs have mismatched shapes or are empty.

---

## Example Usage

```python
from machinegnostics.metrics import root_mean_squared_error

# Example 1: Using lists
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print(root_mean_squared_error(y_true, y_pred))

# Example 2: Using numpy arrays
import numpy as np
y_true = np.array([1, 2, 3])
y_pred = np.array([1, 2, 2])
print(root_mean_squared_error(y_true, y_pred))
```

---

## Notes

- Compatibility: Accepts NumPy arrays and Pandas Series.
- Both `y_true` and `y_pred` must have the same shape and must not be empty.
- RMSE is sensitive to outliers due to the squaring of errors.
- RMSE is in the same units as the target variable, making it easy to interpret.

---

**Author:** Nirmal Parmar		
**Date:** 2025-09-24

---
