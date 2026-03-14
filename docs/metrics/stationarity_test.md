# stationarity_test: Gnostic Stationarity Test

The `stationarity_test` function checks for stationarity in a time series using the homogeneity of Residual Entropy. This method leverages the Machine Gnostics framework to provide a robust assessment of whether the statistical properties of a time series are constant over time.

---

## Overview

This function analyzes the stationarity of a time series by computing the **Residual Entropy (RE)** over a sliding window. It then determines if the sequence of RE values is homogeneous using the **DataHomogeneity** test.

The process involves:

1.  Sliding a window of size `window_size` across the data.
2.  Fitting an Estimation Global Distribution Function (EGDF) to the data in each window.
3.  Extracting the Residual Entropy for each window.
4.  Testing the sequence of entropy values for gnostic homogeneity.

If the Residual Entropy sequence is homogeneous, the time series is considered stationary.

---

## Parameters

| Parameter     | Type                           | Description                                                                 | Default  |
| ------------- | ------------------------------ | --------------------------------------------------------------------------- | -------- |
| `data`        | array-like or Pandas Series    | Time series data to analyze (1D).                                           | Required |
| `window_size` | int                            | Size of the sliding window for entropy calculation.                         | `10`     |
| `S`           | str                            | Scale parameter for EGDF fitting (`'auto'` or float).                       | `'auto'` |
| `data_form`   | str                            | Form of input data: `'a'` (additive) or `'m'` (multiplicative).             | `'a'`    |
| `verbose`     | bool                           | If True, enables detailed logging for debugging.                            | `False`  |

---

## Returns

- **bool**  
  `True` if the time series is stationary (Residual Entropy is homogeneous), `False` otherwise.

---

## Raises

- **TypeError**  
  If `data` is not array-like.
- **ValueError**  
  If data is empty, or if `window_size` is invalid (must be less than data length).

---

## Example Usage

```python
from machinegnostics.metrics import stationarity_test
import numpy as np

# Example 1: Stationary data (Gaussian noise)
data_stationary = np.random.normal(0, 1, 100)
is_stat = stationarity_test(data_stationary, window_size=20)
print(f"Is stationary: {is_stat}")

# Example 2: Non-stationary data (Trend)
t = np.linspace(0, 10, 100)
data_trend = np.sin(t) + t  # Sine wave with linear trend
is_stat_trend = stationarity_test(data_trend, window_size=20)
print(f"Is stationary: {is_stat_trend}")
```

---

## Notes

- **Window Size**: The `window_size` determines the locality of the entropy calculation. It must be smaller than the total length of the data but large enough to fit an EGDF (typically > 3).
- **Residual Entropy**: This metric captures the uncertainty within the local window. If this uncertainty remains consistent (homogeneous) throughout the series, the process is deemed stationary.
- **Robustness**: Like other Gnostic metrics, this test is robust to outliers and does not rely on Gaussian assumptions.
- **Compatibility**: Accepts NumPy arrays and Pandas Series as input.

---

**Author:** Nirmal Parmar  
**Date:** 2026-02-02
