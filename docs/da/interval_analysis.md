# IntervalAnalysis: Marginal Interval Analysis (Machine Gnostics)

The `IntervalAnalysis` class provides robust, adaptive, and diagnostic interval estimation for Gnostic Distribution Functions (GDFs) such as ELDF, EGDF, QLDF, and QGDF. It estimates meaningful data intervals (tolerance, typical intervals) based on the behavior of the GDF's central parameter (Z0) as the data is extended, enforcing ordering constraints and providing detailed diagnostics.

---

## Overview

IntervalAnalysis orchestrates the complete process of fitting GDFs, checking homogeneity, and computing robust data intervals using the DataIntervals engine. It is designed for reliability, diagnostics, and adaptive interval estimation in scientific and engineering data analysis.

**Gnostic vs. Statistical Interval Analysis:**Gnostic interval analysis does not rely on probabilistic or statistical assumptions. Instead, it uses algebraic and geometric properties of the data and distribution functions, providing deterministic, reproducible, and interpretable intervals even for small, noisy, or non-Gaussian datasets. This is fundamentally different from classical statistical interval estimation, which depends on distributional assumptions and sampling theory.

- **Assumption-Free:** No parametric or probabilistic assumptions.
- **Robust:** Handles outliers, heterogeneity, and bounded/unbounded domains.
- **Adaptive:** Intervals adapt to data structure and central parameter behavior.
- **Diagnostics:** Tracks warnings, errors, and intermediate results.
- **Visualization:** Built-in plotting for distributions and intervals.
- **Memory-Efficient:** Optional flushing of intermediate arrays.

---

## Key Features

- End-to-end interval estimation for GDFs
- Automatic homogeneity testing and diagnostics
- Adaptive tolerance and typical interval computation
- Handles weighted, bounded, and unbounded data
- Detailed error and warning logging
- Visualization of fitted distributions and intervals
- Deterministic and reproducible results

---

## Parameters

| Parameter                    | Type               | Default    | Description                                               |
| ---------------------------- | ------------------ | ---------- | --------------------------------------------------------- |
| `DLB`                      | float or None      | None       | Data Lower Bound (absolute minimum, optional)             |
| `DUB`                      | float or None      | None       | Data Upper Bound (absolute maximum, optional)             |
| `LB`                       | float or None      | None       | Lower Probable Bound (practical lower limit, optional)    |
| `UB`                       | float or None      | None       | Upper Probable Bound (practical upper limit, optional)    |
| `S`                        | float or 'auto'    | 'auto'     | Scale parameter for distribution                          |
| `z0_optimize`              | bool               | True       | Optimize central parameter Z0 during fitting              |
| `tolerance`                | float              | 1e-9       | Convergence tolerance for optimization                    |
| `data_form`                | str                | 'a'        | Data form: 'a' (additive), 'm' (multiplicative)           |
| `n_points`                 | int                | 10         | Number of points for distribution evaluation              |
| `n_points_gdf`             | int                | 1000       | Number of points for smooth GDF generation                |
| `homogeneous`              | bool               | True       | Assume data homogeneity (enables homogeneity testing)     |
| `catch`                    | bool               | True       | Store warnings/errors and intermediate results            |
| `weights`                  | array-like or Series or None | None       | Prior weights for data points (accepts NumPy arrays or Pandas Series).                             |
| `wedf`                     | bool               | False      | Use Weighted Empirical Distribution Function              |
| `opt_method`               | str                | 'Powell'   | Optimization method (scipy.optimize)                      |
| `verbose`                  | bool               | False      | Print detailed progress and diagnostics                   |
| `max_data_size`            | int                | 1000       | Max data size for smooth GDF generation                   |
| `flush`                    | bool               | True       | Flush intermediate arrays after fitting                   |
| `dense_zone_fraction`      | float              | 0.4        | Fraction of domain near Z0 for dense interval search      |
| `dense_points_fraction`    | float              | 0.7        | Fraction of search points in dense zone                   |
| `convergence_window`       | int                | 15         | Window size for convergence detection                     |
| `convergence_threshold`    | float              | 1e-6       | Threshold for Z0 convergence                              |
| `min_search_points`        | int                | 30         | Minimum search points before checking convergence         |
| `boundary_margin_factor`   | float              | 0.001      | Margin factor to avoid searching at boundaries            |
| `extrema_search_tolerance` | float              | 1e-6       | Tolerance for detecting extrema in Z0 variation           |
| `gnostic_filter`           | bool               | False      | Apply gnostic clustering to filter outlier Z0 values      |
| `cluster_bounds`           | bool               | True       | Estimate cluster bounds using DataCluster                 |
| `membership_bounds`        | bool               | True       | Estimate membership bounds using DataMembership           |

---

## Attributes

- **params**: `dict`
  Stores all warnings, errors, and diagnostic information from the analysis.

---

## Methods

### `fit(data, plot=False)`

Runs the complete interval analysis workflow on the input data.

- **data**: array-like or Series, shape `(n_samples,)`  
  Input data for interval analysis. Accepts NumPy arrays or Pandas Series/DataFrame column.
- **plot**: `bool` (optional)
  If True, automatically generates diagnostic plots after fitting

**Returns:**
`dict` — Estimated interval bounds and diagnostics

---

### `results()`

Returns a dictionary of estimated interval results and bounds. Also called 'Data Certification'

**Returns:**
`dict` — Contains keys such as `'LB', 'LSB', 'DLB', 'LCB', 'LSD', 'ZL', 'Z0L', 'Z0', 'Z0U', 'ZU', 'USD', 'UCB', 'DUB', 'USB', 'UB'`

  - **LB**: Lower Bound
  The practical lower limit for the interval (may be set by user or inferred).

  - **LSB**: Lower Sample (Membership) Bound
  The lowest value for which data is homogeneous.

  - **DLB**: Data Lower Bound
  The absolute minimum value present in the data.

  - **LCB**: Lower Cluster Bound
  The lower edge of the main data cluster.

  - **LSD**: Lower Standard Deviation Bound
  The lowest value as per gnostic standard deviation.

  - **ZL**: Z0 Lower Interval
  The lower bound of the typical interval.

  - **Z0L**: Z0 Lower Bound
  The lower bound of the tolerance interval.

  - **Z0**: Central Value (Gnostic Mean)
  The central parameter of the distribution (gnostic mean).

  - **Z0U**: Z0 Upper Bound
  The upper bound of the tolerance interval.

  - **ZU**: Z0 Upper Interval
  The upper bound of the typical interval.

  - **USD**: Upper Support/Domain Bound
  The highest value in the support or domain of the fitted distribution.

  - **UCB**: Upper Cluster Bound
  The upper edge of the main data cluster.

  - **DUB**: Data Upper Bound
  The absolute maximum value present in the data.

  - **USB**: Upper Sample (Membership) Bound
  The highest value for which data is homogeneous (membership analysis).

  - **UB**: Upper Bound
  The practical upper limit for the interval (may be set by user or inferred).

---

### `plot(GDF=True, intervals=True)`

Visualizes the fitted GDFs and the estimated intervals.

- **GDF**: `bool` (default: True)Plot the fitted ELDF (local distribution function)
- **intervals**: `bool` (default: True)
  Plot the estimated intervals and Z0 variation

**Returns:**
None (displays plot)

---

## Example Usage

=== "Python"

    ```python
    import numpy as np
    from machinegnostics.magcal import IntervalAnalysis

    # Example data
    data = np.array([ -13.5, 0, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])

    # Initialize IntervalAnalysis
    ia = IntervalAnalysis(verbose=True)

    # Fit and get interval results
    ia.fit(data, plot=True)
    print(ia.results())

    # Visualize results
    ia.plot(GDF=True, intervals=True)
    ```

=== "Output"

    ![Interval Analysis Plot](image/interval_analysis/1770033188023.png)

---

## Notes

- Gnostic interval analysis is fundamentally different from statistical interval analysis: it does not rely on probability or sampling theory, but on algebraic and geometric properties of the data and distribution functions.
- Homogeneity of the data is checked automatically; warnings are issued if violated.
- Suitable for scientific, engineering, and reliability applications.
- All warnings and errors are stored in the `params` attribute for later inspection.

---

**Author:** Nirmal Parmar   
**Date:** 2025-09-24

---
