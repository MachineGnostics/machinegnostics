# median: Gnostic Median Metric

The `median` function computes the Gnostic median (Global Estimate of Location) of a data sample. This metric uses gnostic theory to provide robust, assumption-free estimates of central tendency, leveraging irrelevance and fidelity measures for deeper insight into data structure and uncertainty.

---

## Overview

Gnostic median generalizes classical median by using irrelevance and fidelity measures:

- **Case `'i'`**: Estimates median using EGDF (Empirical Gnostics Distribution Function).
- **Case `'j'`**: Quantifies median using QGDF (Quantile Gnostics Distribution Function).

Both approaches are robust to outliers and non-normal data, providing reliable diagnostics in challenging scenarios.

- Compatibility: Accepts NumPy arrays and Pandas Series for `data`.

---

## Parameters

| Parameter       | Type                           | Description                                                                  | Default   |
| --------------- | ------------------------------ | ---------------------------------------------------------------------------- | --------- |
| `data`          | array-like or Pandas Series    | Input data (1D, no NaN/Inf).                                                 | Required  |
| `case`          | str        | `'i'` for estimating median (EGDF), `'j'` for quantifying median (QGDF).     | `'i'`     |
| `S`             | float/str  | Scaling parameter for EGDF/QGDF. Can be `float` or `'auto'`.                 | `'auto'`  |
| `z0_optimize`   | bool       | Whether to optimize z0 in EGDF/QGDF.                                         | `True`    |
| `data_form`     | str        | Data form for EGDF/QGDF: `'a'` for additive, `'m'` for multiplicative.       | `'a'`     |
| `tolerance`     | float      | Tolerance for EGDF/QGDF fitting.                                             | `1e-6`    |
| `verbose`       | bool       | If True, enables detailed logging for debugging.                             | `False`   |

---

## Returns

- **float**
  The Gnostic median of the data.

---

## Raises

- **TypeError** If input is not array-like (NumPy arrays, lists) or Pandas Series, or if `S` is not a float or `'auto'`.
- **ValueError**
  If input is not 1D, is empty, contains NaN/Inf, or if `case`/`data_form` is invalid.

---

## Example Usage

```python
import machinegnostics as mg
import numpy as np

# Example 1: Compute gnostic median (default case)
data = np.array([1, 2, 3, 4, 5])
median_value = mg.median(data)
print(median_value)

# Example 2: Quantifying median with QGDF
median_j = mg.median(data, case='j')
print(median_j)
```

---

## Notes

- The function uses EGDF or QGDF to compute irrelevance and fidelity values, which are then used to estimate the median.
- Input data must be 1D, cleaned, and free of NaN/Inf.
- The metric is robust to outliers and non-normal data, providing more reliable diagnostics than classical median.
- Scaling (`S`), optimization (`z0_optimize`), and data form (`data_form`) parameters allow for flexible analysis.

---

## Gnostic vs. Classical Median

> **Note:**
> Unlike classical median metrics that use statistical order statistics, the Gnostic median is computed using irrelevance and fidelity measures from gnostic theory. This approach is assumption-free and designed to reveal the true diagnostic properties of your data.

---

**Author:** Nirmal Parmar	
**Date:** 2025-09-24

---
