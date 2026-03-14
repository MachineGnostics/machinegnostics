# std: Gnostic Standard Deviation Metric

The `std` function computes the Gnostic standard deviation of a data sample. This metric uses gnostic theory to provide robust, assumption-free estimates of data dispersion, leveraging irrelevance and fidelity measures for deeper insight into uncertainty and structure.

---

## Overview

Gnostic standard deviation generalizes classical standard deviation by using irrelevance and fidelity measures:

- **Case `'i'`**: Estimates standard deviation using the estimating geometry.
- **Case `'j'`**: Quantifies standard deviation using the quantifying geometry.

Both approaches are robust to outliers and non-normal data, providing reliable diagnostics in challenging scenarios. The function returns lower and upper bounds for the standard deviation.

- Compatibility: Accepts NumPy arrays and Pandas Series for `data`.

---

## Parameters

| Parameter     | Type                           | Description                                                                                   | Default      |
|---------------|--------------------------------|-----------------------------------------------------------------------------------------------|--------------|
| `data`        | array-like or Pandas Series    | Input data (1D, no NaN/Inf).                                                                  | Required     |
| `case`        | str      | `'i'` for estimating standard deviation, `'j'` for quantifying standard deviation.            | `'i'`        |
| `S`           | float/str | Scaling parameter for EGDF. Can be `float` or `'auto'`. Suggested range: [0.01, 10].            | `'auto'`     |
| `z0_optimize` | bool     | Whether to optimize z0 in ELDF.                                                               | `True`       |
| `data_form`   | str      | Data form for ELDF: `'a'` for additive, `'m'` for multiplicative.                             | `'a'`        |
| `tolerance`   | float    | Tolerance for ELDF fitting.                                                                   | `1e-6`       |
| `verbose`     | bool     | If True, enables detailed logging for debugging.                                              | `False`      |

---

## Returns

- **tuple**  
  Lower and upper bounds of the standard deviation range.

---

## Raises

- **TypeError**  
  If input is not array-like (NumPy arrays, lists) or Pandas Series, or if `S` is not a float or `'auto'`.
- **ValueError**  
  If input is not 1D, is empty, contains NaN/Inf, or if `case`/`data_form` is invalid.

---

## Example Usage

```python
import machinegnostics as mg
import numpy as np

# Example 1: Compute gnostic standard deviation (default case)
data = np.array([1, 2, 3, 4, 5])
std_vals = mg.std(data)
print(std_vals)  # Returns (lb, ub)

# Example 2: Quantifying standard deviation
std_vals_j = mg.std(data, case='j')
print(std_vals_j)
```

---

## Notes

- The function uses mean and variance from gnostic theory to compute lower and upper bounds for standard deviation.
- Input data must be 1D, cleaned, and free of NaN/Inf.
- The metric is robust to outliers and non-normal data, providing more reliable diagnostics than classical standard deviation.
- Scaling (`S`), optimization (`z0_optimize`), and data form (`data_form`) parameters allow for flexible analysis.
- If `S='auto'`, the function optimizes the scaling parameter using EGDF.

---

## Gnostic vs. Classical Standard Deviation

> **Note:**  
> Unlike classical standard deviation metrics that use statistical means and variances, the Gnostic standard deviation is computed using irrelevance and fidelity measures from gnostic theory. This approach is assumption-free and designed to reveal the true diagnostic properties of your data.

---

**Authors:** Nirmal Parmar  
**Date:** 2025-09-24

---
