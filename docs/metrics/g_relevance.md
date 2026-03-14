# hc: Gnostic Characteristics (Hc) Metric

The `hc` function computes the Gnostic Characteristics (Hc) metric for a set of true and predicted values. This metric evaluates the relevance or irrelevance of predictions using gnostic theory, providing robust, assumption-free diagnostics for model performance.

---

## Overview

The Hc metric measures the gnostic relevance or irrelevance between true and predicted values:

- **Case `'i'`**: Estimates gnostic relevance. Values close to one indicate less relevance. Range: [0, 1].
- **Case `'j'`**: Estimates gnostic irrelevance. Values close to 1 indicate less irrelevance. Range: [0, ∞).

Unlike classical metrics, Hc uses gnostic algebra to provide deeper insight into the relationship between predictions and actual outcomes, especially in the presence of outliers or non-normal data.

- Compatibility: Accepts NumPy arrays and Pandas Series for `y_true` and `y_pred`.

---

## Parameters

| Parameter   | Type                           | Description                                                         |
| ----------- | ------------------------------ | ------------------------------------------------------------------- |
| `y_true`    | array-like or Pandas Series    | True values (1D).                                                   |
| `y_pred`    | array-like or Pandas Series    | Predicted values (1D).                                              |
| `case`      | str                            | `'i'` for relevance, `'j'` for irrelevance. Default: `'i'`.         |
| `S`         | float/str                      | Gnostic scale parameter. Default: `'auto'`.                         |
| `verbose`   | bool                           | If True, enables detailed logging for debugging. Default: `False`.  |

---

## Returns

- **float**
  The calculated Hc value (mean of squared gnostic characteristics).

---

## Raises

- **TypeError** If `y_true` or `y_pred` are not array-like (NumPy arrays, lists) or Pandas Series.
- **ValueError**
  If inputs are empty, contain NaN/Inf, are not 1D, have mismatched shapes, or if `case` is not `'i'` or `'j'`.

---

## Example Usage

```python
from machinegnostics.metrics import hc

# Example 1: Using lists
y_true = [1, 2, 3]
y_pred = [1, 2, 3]
hc_value = hc(y_true, y_pred, case='i')
print(hc_value)

# Example 2: Using numpy arrays and irrelevance case
import numpy as np
y_true = np.array([2, 4, 6])
y_pred = np.array([1, 2, 3])
hc_value = hc(y_true, y_pred, case='j', verbose=True)
print(hc_value)
```

---

## Notes

- The function supports input as lists, tuples, or numpy arrays.
- Both `y_true` and `y_pred` must be 1D, have the same shape, and must not be empty or contain NaN/Inf.
- For standard comparison, irrelevances are calculated with S=1.
- The Hc metric is robust to outliers and non-normal data, providing more reliable diagnostics than classical metrics.

---

## Gnostic vs. Classical Metrics

> **Note:**
> Unlike traditional metrics that use statistical means, the Hc metric is computed using gnostic algebra and characteristics. This approach is assumption-free and designed to reveal the true diagnostic properties of your data and model predictions.

---

**Author:** Nirmal Parmar
**Date:** 2025-09-24

---
