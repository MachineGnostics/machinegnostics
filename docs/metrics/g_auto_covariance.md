# auto_covariance: Gnostic Auto-Covariance Metric

The `auto_covariance` function computes the Gnostic auto-covariance coefficient for a data sample. This metric measures the relationship between a data sample and a lagged version of itself, using robust gnostic theory principles for reliable diagnostics—even in the presence of noise or outliers.

---

## Overview

Auto-covariance quantifies how much a data sample co-varies with itself when shifted by a specified lag.
Unlike classical auto-covariance, the Gnostic version uses irrelevance measures from gnostic theory, providing robust, assumption-free estimates that reflect the true structure of your data.

- Compatibility: Accepts NumPy arrays and Pandas Series for `data`.

---

## Parameters

| Parameter   | Type                           | Description                                                                                    |
| ----------- | ------------------------------ | ---------------------------------------------------------------------------------------------- |
| `data`    | array-like or Pandas Series     | Data sample (1D, no NaN/Inf). Represents a time series or sequential data.                    |
| `lag`     | int                            | Lag value (non-negative, less than length of data). Default:`0`.                               |
| `case`    | str                            | Geometry type:`'i'` for estimation (EGDF), `'j'` for quantifying (QGDF). Default: `'i'`.       |
| `verbose` | bool                           | If True, enables detailed logging for debugging. Default:`False`.                              |

---

## Returns

- **float**
  The Gnostic auto-covariance coefficient for the given lag. If the computed value is less than 1e-6, it is set to 0.0.

---

## Raises

- **ValueError**
  If input is not 1D array-like, is empty, contains NaN/Inf, or if lag/case is invalid.

---

## Example Usage

```python
import numpy as np
from machinegnostics.metrics import auto_covariance

# Example 1: Compute auto-covariance for a simple dataset
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
lag = 1
auto_covar = auto_covariance(data, lag=lag, case='i')
print(f"Auto-covariance with lag={lag}: {auto_covar}")

# Example 2: Compute auto-covariance with QGDF
auto_covar_j = auto_covariance(data, lag=2, case='j')
print(f"Auto-covariance with lag=2: {auto_covar_j}")

# Example 3: Handle invalid input
data_invalid = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
try:
    auto_covar = auto_covariance(data_invalid, lag=1, case='i')
except ValueError as e:
    print(f"Error: {e}")
```

---

## Notes

- The function uses Gnostic theory to compute irrelevance values for the data and its lagged version.
- Irrelevance values are clipped to avoid overflow, with a maximum value of 1e12.
- Homogeneity checks are performed on the data and its lagged version. If the data is not homogeneous, warnings are raised and scale parameters are adjusted.
- The metric is robust to data uncertainty, noise, and outliers.
- Input data must be preprocessed and cleaned for optimal results.

---

## Gnostic vs. Classical Auto-Covariance

> **Note:**
> Unlike classical auto-covariance metrics that rely on statistical means, the Gnostic auto-covariance uses irrelevance measures derived from gnostic theory. This approach is assumption-free and designed to reveal true temporal relationships, even in the presence of outliers or non-normal distributions.

---

**Author:** Nirmal Parmar
**Date:** 2025-09-24

---