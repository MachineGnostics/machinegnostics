# mean: Gnostic Mean Metric

The `mean` function computes the Gnostic mean (Local Estimate of Location) of a data sample. This metric uses gnostic theory to provide robust, assumption-free estimates of central tendency, leveraging irrelevance and fidelity measures for deeper insight into data structure and uncertainty.

---

## Overview

Gnostic mean generalizes classical mean by using irrelevance and fidelity measures:

- **Case `'i'`**: Estimates mean using ELDF (Empirical Likelihood Distribution Function).
- **Case `'j'`**: Quantifies mean using QLDF (Quantile Likelihood Distribution Function).

Both approaches are robust to outliers and non-normal data, providing reliable diagnostics in challenging scenarios.

- Compatibility: Accepts NumPy arrays and Pandas Series for `data`.

---

## Parameters

| Parameter       | Type                           | Description                                                               | Default   |
| --------------- | ------------------------------ | ------------------------------------------------------------------------- | --------- |
| `data`          | array-like or Pandas Series    | Input data (1D, no NaN/Inf).                                              | Required  |
| `S`             | float/str                      | Scaling parameter (`float` or `'auto'`). Suggested: [0.01, 2].            | `'auto'`  |
| `case`          | str                            | `'i'` for estimating mean (ELDF), `'j'` for quantifying mean (QLDF).      | `'i'`     |
| `z0_optimize`   | bool                           | Whether to optimize z0.                                                   | `True`    |
| `data_form`     | str                            | Data form: `'a'` for additive, `'m'` for multiplicative.                  | `'a'`     |
| `tolerance`     | float                          | Tolerance for ELDF/QLDF fitting.                                          | `1e-6`    |
| `verbose`       | bool                           | If True, enables detailed logging for debugging.                          | `False`   |

---

## Returns

- **float**
  The Gnostic mean of the data.

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

# Example 1: Compute gnostic mean (default case)
data = np.array([1, 2, 3, 4, 5])
mean_value = mg.mean(data)
print(mean_value)

# Example 2: Quantifying mean with QLDF
mean_j = mg.mean(data, case='j')
print(mean_j)
```

---

## Notes

- The function uses ELDF or QLDF to compute irrelevance and fidelity values, which are then used to estimate the mean.
- Input data must be 1D, cleaned, and free of NaN/Inf.
- The metric is robust to outliers and non-normal data, providing more reliable diagnostics than classical mean.
- Scaling (`S`), optimization (`z0_optimize`), and data form (`data_form`) parameters allow for flexible analysis.

---

## Gnostic vs. Classical Mean

> **Note:**
> Unlike classical mean metrics that use statistical averages, the Gnostic mean is computed using irrelevance and fidelity measures from gnostic theory. This approach is assumption-free and designed to reveal the true diagnostic properties of your data.

---

**Authors:** Nirmal Parmar	
**Date:** 2025-09-24

---
