# mean_squared_error: Mean Squared Error (MSE) Metric

The `mean_squared_error` function computes the mean squared error (MSE) between true and predicted values. MSE is a fundamental regression metric that measures the average of the squares of the errors—that is, the average squared difference between the estimated values and the actual value.

> Unlike traditional error metrics that use the statistical mean, Machine Gnostics metrics are computed using the gnostic mean. The gnostic mean is a robust, assumption-free measure designed to provide deeper insight and reliability, especially in the presence of outliers or non-normal data.
> This approach ensures that error metrics reflect the true structure and diagnostic properties of your data, in line with the principles of Mathematical Gnostics.

---

## Overview

Mean Squared Error is defined as the average of the squared differences between actual and predicted values.

MSE is widely used in regression analysis to quantify the accuracy of predictions. Lower MSE values indicate better model performance, while higher values indicate larger errors.

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
  The average of squared differences between actual and predicted values.

---

## Raises

- **TypeError** If `y_true` or `y_pred` are not array-like (NumPy arrays, lists) or Pandas Series.
- **ValueError** If inputs have mismatched shapes or are empty.

---

## Example Usage

```python
from machinegnostics.metrics import mean_squared_error

# Example 1: Using lists
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print(mean_squared_error(y_true, y_pred))

# Example 2: Using numpy arrays
import numpy as np
y_true = np.array([1, 2, 3])
y_pred = np.array([1, 2, 2])
print(mean_squared_error(y_true, y_pred))
```

---

## Notes

- Compatibility: Accepts NumPy arrays and Pandas Series.
- Both `y_true` and `y_pred` must have the same shape and must not be empty.
- MSE penalizes larger errors more than MAE (mean absolute error), making it sensitive to outliers.

---

**Author:** Nirmal Parmar		
**Date:** 2025-09-24

---
