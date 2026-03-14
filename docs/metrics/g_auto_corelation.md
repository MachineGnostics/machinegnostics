# auto_correlation: Gnostic Auto-Correlation Metric

The `auto_correlation` function computes the Gnostic auto-correlation coefficient for a data sample. This metric measures the similarity between a data sample and a lagged version of itself, using robust gnostic theory principles for reliable diagnostics—even in the presence of noise or outliers.

---

## Overview

Auto-correlation quantifies how much a data sample resembles itself when shifted by a specified lag.
Unlike classical auto-correlation, the Gnostic version uses irrelevance measures from gnostic theory, providing robust, assumption-free estimates that reflect the true structure of your data.

- Compatibility: Accepts NumPy arrays and Pandas Series for `data`.

---

## Parameters

| Parameter   | Type                           | Description                                                                                    |
| ----------- | ------------------------------ | ---------------------------------------------------------------------------------------------- |
| `data`    | array-like or Pandas Series     | Data sample (1D, no NaN/Inf).                                                                  |
| `lag`     | int                            | Lag value (non-negative, less than length of data). Default:`0`.                               |
| `case`    | str                            | Geometry type:`'i'` for estimation (EGDF), `'j'` for quantifying (QGDF). Default: `'i'`.       |
| `verbose` | bool                           | If True, enables detailed logging for debugging. Default:`False`.                              |

---

## Returns

- **float**
  The Gnostic auto-correlation coefficient for the given lag.

---

## Raises

- **ValueError**
  If input is not 1D array-like, is empty, contains NaN/Inf, or if lag/case is invalid.

---

## Example Usage

```python
import numpy as np
from machinegnostics.metrics import auto_correlation

# Example 1: Compute auto-correlation for a simple dataset
data = np.array([1, 2, 3, 4, 5])
lag = 1
auto_corr = auto_correlation(data, lag=lag, case='i', verbose=False)
print(f"Auto-Correlation (lag={lag}, case='i'): {auto_corr}")

# Example 2: Using quantifying geometry
auto_corr_j = auto_correlation(data, lag=2, case='j', verbose=True)
print(f"Auto-Correlation (lag=2, case='j'): {auto_corr_j}")
```

---

## Notes

- The metric is robust to data uncertainty, noise, and outliers.
- Input data must be preprocessed and cleaned for optimal results.
- If data homogeneity is not met, the function automatically adjusts scale parameters for better reliability.
- The Gnostic auto-correlation uses irrelevance measures rather than classical means, providing deeper insight into temporal relationships in your data.
- Supports both estimation and quantification geometries for flexible analysis.

---

## Gnostic vs. Classical Auto-Correlation

> **Note:**
> Unlike classical auto-correlation metrics that rely on statistical means, the Gnostic auto-correlation uses irrelevance measures derived from gnostic theory. This approach is assumption-free and designed to reveal true temporal relationships, even in the presence of outliers or non-normal distributions.

---

**Author:** Nirmal Parmar   
**Date:** 2025-09-24

---
