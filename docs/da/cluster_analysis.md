# ClusterAnalysis: End-to-End Clustering-Based Bound Estimation (Machine Gnostics)

The `ClusterAnalysis` class provides a robust, automated workflow for estimating main cluster bounds in a dataset using Gnostic Distribution Functions (GDFs) and advanced clustering analysis. It is designed for interpretable, reproducible interval estimation in scientific, engineering, and data science applications.

---

## Overview

ClusterAnalysis orchestrates the entire process of fitting a GDF (ELDF/EGDF), assessing data homogeneity, performing cluster boundary detection, and returning interpretable lower and upper cluster bounds (LCB, UCB) for the main data cluster.

- **Automated Pipeline:** Integrates GDF fitting, homogeneity testing, and cluster analysis.
- **Flexible:** Supports both local (ELDF) and global (EGDF) GDFs.
- **Robust:** Handles weighted data, bounded/unbounded domains, and advanced parameterization.
- **Diagnostics:** Detailed error/warning logging and reproducible parameter tracking.
- **Memory-Efficient:** Optional flushing of intermediate results.
- **Visualization:** Built-in plotting for GDF and cluster analysis results.

---

## Key Features

- End-to-end cluster-based bound estimation
- Integrates GDF fitting, homogeneity testing, and clustering
- Supports local and global GDFs
- Handles weighted, bounded, and unbounded data
- Detailed error and warning logging
- Memory-efficient operation via flushing
- Visualization of GDF and cluster analysis results

---

## Parameters

| Parameter              | Type                  | Default   | Description                                                      |
|------------------------|-----------------------|-----------|------------------------------------------------------------------|
| `verbose`              | bool                  | False     | Print detailed logs and progress information                      |
| `catch`                | bool                  | True      | Store intermediate results and diagnostics                        |
| `derivative_threshold` | float                 | 0.01      | Threshold for derivative-based cluster boundary detection         |
| `DLB`                  | float or None         | None      | Data Lower Bound (absolute minimum, optional)                     |
| `DUB`                  | float or None         | None      | Data Upper Bound (absolute maximum, optional)                     |
| `LB`                   | float or None         | None      | Lower probable bound (optional)                                   |
| `UB`                   | float or None         | None      | Upper probable bound (optional)                                   |
| `S`                    | float or 'auto'       | 'auto'    | Scale parameter for GDF ('auto' for automatic estimation)         |
| `varS`                 | bool                  | False     | Use variable scale parameter during optimization                  |
| `z0_optimize`          | bool                  | True      | Optimize location parameter Z0 during fitting                     |
| `tolerance`            | float                 | 1e-5      | Convergence tolerance for optimization                            |
| `data_form`            | str                   | 'a'       | Data form: 'a' (additive), 'm' (multiplicative)                  |
| `n_points`             | int                   | 1000      | Number of points for GDF evaluation                              |
| `homogeneous`          | bool                  | True      | Assume data homogeneity                                          |
| `weights`              | np.ndarray or None    | None      | Prior weights for data points                                    |
| `wedf`                 | bool                  | False     | Use Weighted Empirical Distribution Function                     |
| `opt_method`           | str                   | 'L-BFGS-B'| Optimization method (scipy.optimize)                             |
| `max_data_size`        | int                   | 1000      | Max data size for smooth GDF generation                          |
| `flush`                | bool                  | False     | Flush intermediate results after fitting to save memory           |

---

## Attributes

- **LCB**: `float or None`  
  Lower Cluster Bound (main cluster lower edge)
- **UCB**: `float or None`  
  Upper Cluster Bound (main cluster upper edge)
- **params**: `dict`  
  All parameters, intermediate results, errors, and warnings
- **_fitted**: `bool`  
  Indicates whether analysis has been completed

---

## Methods

### `fit(data, plot=False)`

Runs the full cluster analysis pipeline on the input data.

- **data**: array-like or Series, shape `(n_samples,)`  
  Input data for interval analysis. Accepts NumPy arrays or Pandas Series/DataFrame column.
- **plot**: `bool` (optional)  
  If True, generates plots for the fitted GDF and cluster analysis

**Returns:**  
`tuple` — `(LCB, UCB)` as the main cluster bounds

---

### `results()`

Returns a dictionary with the estimated bounds and key results.

**Returns:**  
`dict` — `{ 'LCB': float, 'UCB': float }`

---

### `plot()`

Visualizes the fitted GDF and cluster analysis results (if not flushed).

**Returns:**  
None (displays plot)

---

## Example Usage

=== "Python"

    ```python
    import numpy as np
    from machinegnostics.magcal import ClusterAnalysis

    # Example data
    data = np.array([ -13.5, 0, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])

    # Initialize ClusterAnalysis
    ca = ClusterAnalysis(verbose=True)

    # Fit and get cluster bounds
    LCB, UCB = ca.fit(data)
    print(f"Main cluster bounds: LCB={LCB:.3f}, UCB={UCB:.3f}")

    # Visualize results
    ca.plot()

    # Access results dictionary
    results = ca.results()
    print(results)
    ```

=== "Output"

    ![Cluster Analysis Plot](image/cluster_analysis/1770032920673.png)

---

## Notes

- Designed for robust, interpretable cluster-based bound estimation
- Works best with local GDFs (ELDF); global GDFs (EGDF) are supported
- If `homogeneous=True` but data is heterogeneous, a warning is issued
- All intermediate parameters, errors, and warnings are tracked in `params`
- For large datasets or memory-constrained environments, set `flush=True` to save memory (disables plotting)

---


**Author:** Nirmal Parmar  
**Date:** 2025-09-24

---