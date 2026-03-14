# DataIntervals: Robust Interval Estimation Engine (Machine Gnostics)

The `DataIntervals` class provides robust, adaptive, and diagnostic interval estimation for Gnostics Distribution Function (GDF) classes such as ELDF, EGDF, QLDF, and QGDF. It is designed to estimate meaningful data intervals (such as tolerance and typical intervals) based on the behavior of the GDF's central parameter (Z0) as the data is extended, while enforcing ordering constraints and providing detailed diagnostics.

---

## Overview

DataIntervals is designed for advanced interval analysis in scientific, engineering, and reliability applications. It adaptively scans the data domain, extending the data with candidate values and tracking the variation of the central parameter Z0.

**Key Features:**

- **Adaptive Search**: Efficiently scans the data domain with a dense search near the central value (Z0) and sparser search near the boundaries.
- **Robustness**: Supports gnostic filtering (clustering) to enhance robustness against outliers.
- **Diagnostics**: Provides warnings and errors for suboptimal settings and ordering violations.
- **Ordering Constraint**: Ensures that the estimated intervals satisfy the natural ordering: ZL < Z0L < Z0 < Z0U < ZU.
- **Visualization**: Plots Z0 variation, estimated intervals, and data coverage.

---

## Parameters

| Parameter                  | Type                | Default | Description                                                     |
| -------------------------- | ------------------- | ------- | --------------------------------------------------------------- |
| `gdf`                      | ELDF/EGDF/QLDF/QGDF | -       | Fitted GDF (Gnostics Distribution Function) object.             |
| `n_points`                 | int                 | 10      | Number of search points for interval estimation.                |
| `dense_zone_fraction`      | float               | 0.4     | Fraction of the search domain near Z0 to search densely.        |
| `dense_points_fraction`    | float               | 0.7     | Fraction of points allocated to the dense zone.                 |
| `convergence_window`       | int                 | 15      | Number of points in the moving window for convergence.          |
| `convergence_threshold`    | float               | 1e-6    | Threshold for standard deviation of Z0 in convergence window.   |
| `min_search_points`        | int                 | 30      | Minimum number of search points before checking convergence.    |
| `boundary_margin_factor`   | float               | 0.001   | Margin factor to avoid searching exactly at the boundaries.     |
| `extrema_search_tolerance` | float               | 1e-6    | Tolerance for detecting extrema in Z0 variation.                |
| `gnostic_filter`           | bool                | False   | If True, apply gnostic clustering to filter outlier Z0 values.  |
| `catch`                    | bool                | True    | If True, catch and store warnings/errors internally.            |
| `verbose`                  | bool                | False   | If True, print detailed progress and diagnostics.               |
| `flush`                    | bool                | False   | If True, flush memory after fitting to save resources.          |

---

## Attributes

- **ZL**: `float`
  Lower bound of the typical data interval.
- **Z0L**: `float`
  Lower bound of the tolerance interval (Z0-based).
- **Z0**: `float`
  Central value (Z0) of the original GDF.
- **Z0U**: `float`
  Upper bound of the tolerance interval (Z0-based).
- **ZU**: `float`
  Upper bound of the typical data interval.
- **tolerance_interval**: `float`
  Width of the tolerance interval (Z0U - Z0L).
- **typical_data_interval**: `float`
  Width of the typical data interval (ZU - ZL).
- **ordering_valid**: `bool`
  Whether the ordering constraint (ZL < Z0L < Z0 < Z0U < ZU) is satisfied.
- **params**: `dict`
  Dictionary of parameters, warnings, errors, and results.
- **search_results**: `dict`
  Raw search results for datum values and corresponding Z0s.

---

## Methods

### `fit(plot=False)`

Run the interval estimation process. Optionally plot results.

- **plot**: `bool` (default=False). If True, automatically plot analysis results.

**Returns:**
None

---

### `results()`

Return a dictionary of estimated interval results and bounds.

**Returns:**
`Dict` — A dictionary containing keys for LB, LSB, DLB, LCB, LSD, ZL, Z0L, Z0, Z0U, ZU, USD, UCB, DUB, USB, UB.

---

### `plot_intervals(figsize=(12, 8))`

Plot the Z0 variation and estimated intervals.

- **figsize**: `tuple` (default=(12, 8)).

**Returns:**
None

---

### `plot(figsize=(12, 8))`

Plot the GDF, PDF, and intervals on the data domain.

- **figsize**: `tuple` (default=(12, 8)).

**Returns:**
None

---

## Example Usage

=== "Python"

    ```python
    import numpy as np
    from machinegnostics.magcal import ELDF, DataIntervals

    data = np.array([-13.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    # 1. Fit ELDF first
    eld = ELDF()
    eld.fit(data)
    
    # 2. Run Interval Analysis
    # create a DataIntervals object
    data_intervals = DataIntervals(gdf=eldf)

    # fit the data
    data_intervals.fit()

    # plot
    data_intervals.plot()

    # print intervals
    data_intervals.results()
    ```

=== "Output"

    ![1770037703442](image/data_interval/1770037703442.png)

---

## Notes

- **GDF Types**: For best results, use with ELDF or QLDF.
- **Optimization**: increasing `n_points` improves accuracy but increases computation time.
- **Usage**: The class is designed for research and diagnostic use; adjust parameters for your data and application.

---

**Author:** Nirmal Parmar   
**Date:** 2025-10-10