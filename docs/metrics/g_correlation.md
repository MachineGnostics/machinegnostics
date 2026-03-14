# correlation: Gnostic Correlation Metric

The `correlation` function computes the Gnostic correlation coefficient between two data samples. This metric provides a robust, assumption-free measure of association, leveraging Mathematical Gnostics to deliver reliable results even in the presence of noise, outliers, or non-Gaussian data.

---

## Overview

The Gnostic correlation metric measures the association between two arrays using either estimation geometry (EGDF) or quantifying geometry (QGDF):

- **Case `'i'`**: Uses estimation geometry (EGDF) for correlation.
- **Case `'j'`**: Uses quantifying geometry (QGDF) for correlation.

Unlike classical correlation coefficients, the Gnostic approach is resilient to data uncertainty and is meaningful for both small and large datasets.

- Compatibility: Accepts NumPy arrays and Pandas Series; for `X`, a Pandas DataFrame column is supported.

---

## Parameters

| Parameter | Type                                   | Description                                                                 |
| --------- | -------------------------------------- | --------------------------------------------------------------------------- |
| `X`       | array-like or Pandas Series/DataFrame  | Feature data sample (1D Series or single column from DataFrame, no NaN/Inf).|
| `y`       | array-like or Pandas Series            | Target data sample (1D, no NaN/Inf).                                        |
| `case`    | str                                    | `'i'` for estimation geometry, `'j'` for quantifying geometry. Default: `'i'`. |
| `S`       | str                                    | Gnostic scale parameter. If `'auto'`, determines best scale based on homogeneity. Default: `'auto'`. |
| `verbose` | bool                                   | If True, enables detailed logging for debugging. Default: `False`.           |

---

## Returns

- **float**  
  The Gnostic correlation coefficient between the two data samples.

---

## Raises

- **TypeError**  
  If `X` or `y` are not array-like.
- **ValueError**  
  If input arrays are empty, contain NaN/Inf, are not 1D, have mismatched shapes, or if `case` is not `'i'` or `'j'`.

---

## Example Usage

```python
from machinegnostics.metrics import correlation
import numpy as np

# Example 1: Simple 1D arrays
X = np.array([1, 2, 3, 4, 5])
y = np.array([5, 4, 3, 2, 1])
corr = correlation(X, y, case='i')
print(corr)

# Example 2: Multi-column X
X = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]])
y = np.array([5, 4, 3, 2, 1])
for i in range(X.shape[1]):
    corr = correlation(X[:, i], y)
    print(f"Correlation for column {i}: {corr}")
```

---

## Notes

- Both `X` and `y` must be 1D arrays/Series of the same length, with no missing or infinite values.
- For multi-column `X`, pass each column separately (e.g., `X[:, i]` or a single DataFrame column).
- The metric is robust to outliers and provides meaningful estimates even for small or noisy datasets.
- If data homogeneity is not met, the function will adjust parameters and issue a warning for best results.
- For optimal results, ensure your data is preprocessed and cleaned.

---

## Gnostic vs. Classical Correlation

> **Note:**
> Unlike classical correlation metrics that rely on statistical means and linear relationships, the Gnostic correlation uses irrelevance measures derived from gnostic theory. This approach is assumption-free and designed to reveal true data relationships, even in the presence of outliers or non-normal distributions.

---

**Author:** Nirmal Parmar  
**Date:** 2025-09-24

---