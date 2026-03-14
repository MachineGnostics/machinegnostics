# EGDF: Estimating Global Distribution Function (Machine Gnostics)

The `EGDF` class provides robust, assumption-free global distribution estimation for real-world data using the Machine Gnostics framework. Unlike traditional parametric models, EGDF adapts directly to your data, making it ideal for noisy, uncertain, or heterogeneous datasets.

---

## Overview

EGDF is designed for robust probability and density estimation, especially when data may contain outliers, inner noise, or unknown distributions. It leverages gnostic algebra and error geometry to deliver resilient, interpretable results without requiring prior statistical assumptions.

- **Assumption-Free:** No parametric forms or distributional assumptions.
- **Robust:** Handles outliers, inner noise, and contaminated data.
- **Flexible:** Supports additive and multiplicative data forms.
- **Weighted Data:** Incorporates sample weights for advanced analysis.
- **Automatic Parameter Estimation:** Scale and bounds inferred from data.
- **Memory-Efficient:** Optimized for large datasets.
- **Visualization:** Built-in plotting for EGDF and PDF.
- **Customizable:** Multiple solver options, bounds, and precision settings.

---

## Key Features

- Fits a global distribution function to your data
- Robust to outliers and inner noise
- Supports weighted and unweighted samples
- Automatic or manual bounds and scale selection
- Additive ('a') and multiplicative ('m') data forms
- Advanced optimization with customizable tolerance and solver
- Visualization of EGDF, PDF, and bounds
- Memory-efficient for large datasets
- Detailed results and diagnostics

---

## Parameters

| Parameter         | Type                  | Default   | Description                                                      |
| ----------------- | ---------------------| --------- | ---------------------------------------------------------------  |
| `DLB`             | float or None         | None      | Data Lower Bound (absolute minimum, optional)                    |
| `DUB`             | float or None         | None      | Data Upper Bound (absolute maximum, optional)                    |
| `LB`              | float or None         | None      | Lower Probable Bound (practical lower limit, optional)           |
| `UB`              | float or None         | None      | Upper Probable Bound (practical upper limit, optional)           |
| `S`               | float or 'auto'       | 'auto'    | Scale parameter (auto-estimated or fixed value)                  |
| `z0_optimize`     | bool                  | True      | Optimize location parameter during fitting                       |
| `tolerance`       | float                 | 1e-9      | Convergence tolerance for optimization                           |
| `data_form`       | str                   | 'a'       | Data form: 'a' (additive), 'm' (multiplicative)                  |
| `n_points`        | int                   | 500       | Number of points for distribution curve                          |
| `homogeneous`     | bool                  | True      | Assume data homogeneity                                          |
| `catch`           | bool                  | True      | Store intermediate results (memory usage)                        |
| `weights`         | array-like or Series or None    | None      | Prior weights for data points (accepts NumPy arrays or Pandas Series). |
| `wedf`            | bool                  | False     | Use Weighted Empirical Distribution Function                     |
| `opt_method`      | str                   | 'Powell'  | Optimization method (scipy.optimize)                             |
| `verbose`         | bool                  | False     | Print progress and diagnostics                                   |
| `max_data_size`   | int                   | 1000      | Max data size for smooth EGDF generation                         |
| `flush`           | bool                  | True      | Flush large arrays (memory management)                           |

---

## Attributes

- **params**: `dict`  
  Fitted parameters and results after fitting.
- **DLB, DUB, LB, UB, S, z0_optimize, tolerance, data_form, n_points, homogeneous, catch, weights, wedf, opt_method, verbose, max_data_size, flush**:  
  Configuration parameters as set at initialization.

---

## Methods

### `fit(data, plot=False)`

Fits the EGDF to your data, estimating all relevant parameters and generating the global distribution function.

- **data**: array-like or Series, shape `(n_samples,)`  
  Input data. Accepts NumPy arrays or Pandas Series/DataFrame column.
- **plot**: `bool` (optional)  
  If True, automatically plots the fitted distribution.

**Returns:**  
None (results stored in `params`)

---

### `plot(plot_smooth=True, plot='both', bounds=True, extra_df=True, figsize=(12,8))`

Visualizes the fitted EGDF and related plots.

- **plot_smooth**: `bool`  
  Plot smooth interpolated curve.
- **plot**: `str`  
  'gdf', 'pdf', or 'both'.
- **bounds**: `bool`  
  Show bound lines.
- **extra_df**: `bool`  
  Include additional distribution functions.
- **figsize**: `tuple`  
  Figure size.

**Returns:**  
None (displays plot)

---

### `results()`

Returns a dictionary of all fitted parameters and results.

**Returns:**  
`dict` (fitted parameters, bounds, scale, diagnostics, etc.)

---

## Example Usage

=== "Python"

    ```python
    import numpy as np
    from machinegnostics.magcal import EGDF

    # Example data
    data = np.array([ -13.5, 0, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])

    # Initialize EGDF
    egdf = EGDF()

    # Fit the model
    egdf.fit(data)

    # Plot the results
    egdf.plot()

    # Access fitted parameters
    results = egdf.results()
    print("Global scale parameter:", results['S_opt'])
    print("Distribution bounds:", results['LB'], results['UB'])
    ```

=== "Output"

    ![EGDF Plot](image/egdf/1770031196721.png)

---

## Notes

- EGDF is robust to outliers and suitable for non-Gaussian, contaminated, or uncertain data.
- Supports both additive and multiplicative data forms.
- Use weights for advanced analysis (e.g., reliability, risk).
- For large datasets, set `catch=False` to save memory.
- Visualization options allow in-depth analysis of distribution structure.
- For more information, see [GDF documentation](../mg/gdf.md).
---

**Author:** Nirmal Parmar   
**Date:** 2025-09-24

---