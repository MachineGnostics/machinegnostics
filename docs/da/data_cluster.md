# DataCluster: Advanced Cluster Boundary Detection (Machine Gnostics)

The `DataCluster` class identifies main cluster boundaries (LCB and UCB) from probability density functions (PDF) of fitted Gnostic Distribution Functions (GDFs), including ELDF, EGDF, QLDF, and QGDF.

---

## Overview

DataCluster performs advanced cluster analysis on fitted GDF objects. It uses a **unified derivative-based method** on normalized PDF data to precisely locate the boundaries of the main data cluster.

**Key Features:**

- **Unified Algorithm**: Apply consistent boundary detection across all GDF types (ELDF, EGDF, QLDF, QGDF).
- **PDF Normalization**: Analyzes min-max normalized PDF values [0,1] for robust thresholding.
- **Derivative Analysis**: Locates boundaries where the combined signal (PDF + 1st Derivative) drops below a threshold.
- **Robust Fallback**: Automatically falls back to data bounds if boundary detection fails.
- **Diagnostic Plotting**: Visualizes the PDF, derivative signals, and detected boundaries.

---

## Parameters

| Parameter              | Type                          | Default | Description                                                               |
| ---------------------- | ----------------------------- | ------- | ------------------------------------------------------------------------- |
| `gdf`                  | ELDF/EGDF/QLDF/QGDF           | -       | Fitted GDF object (must have `pdf_points` via `catch=True`).              |
| `verbose`              | bool                          | False   | Enable detailed progress reporting and diagnostic output.                 |
| `catch`                | bool                          | True    | Enable error catching and graceful degradation.                           |
| `derivative_threshold` | float                         | 0.01    | Threshold for boundary detection (lower = wider cluster, higher = narrower). |

---

## Attributes

- **LCB**: `float` or `None`
  Cluster Lower Boundary (left edge of the main cluster).
- **UCB**: `float` or `None`
  Cluster Upper Boundary (right edge of the main cluster).
- **z0**: `float` or `None`
  Characteristic point of the distribution (from GDF).
- **S_opt**: `float` or `None`
  Optimal scale parameter (from GDF).
- **pdf_normalized**: `ndarray` or `None`
  Min-max normalized PDF values [0,1] used for analysis.
- **params**: `dict`
  Complete analysis results including boundaries, methods used, and diagnostics.

---

## Methods

### `fit(plot=False)`

Perform cluster boundary detection analysis.

- **plot**: `bool` (default=False). If True, generates a plot of the results.

**Returns:**
`Tuple[float, float]` — The detected (LCB, UCB) values. Returns `(None, None)` if analysis fails.

---

### `results()`

Return comprehensive analysis results dictionary.

**Returns:**
`Dict` — Dictionary containing:

- Cluster Bound results: `LCB`, `UCB`, `cluster_width`, `clustering_successful`
- GDF Info: `gdf_type`, `Z0`, `S_opt`
- Diagnostics: `method_used`, `normalization_method`, `errors`, `warnings`

---

### `plot(figsize=(12, 8))`

Visualize PDF, boundaries, and derivative analysis.

- **figsize**: `tuple` (default=(12, 8)).

**Returns:**
None. (Displays a two-panel plot: Top=Original PDF, Bottom=Derivative Analysis).

---

## Example Usage

=== "Python"

    ```python
    import numpy as np
    from machinegnostics.magcal import QLDF, DataCluster

    data = np.array([-13.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    # 1. Fit GDF (e.g., ELDF) - Ensure catch=True for PDF points
    # create an ELDF object, outlier robust
    eldf = ELDF()

    # fit the data
    eldf.fit(data=data)

    # 2. Perform Cluster Analysis
    cluster = DataCluster(gdf=eldf, verbose=True)
    CLB, CUB = cluster.fit(plot=True)
    
    if CLB is not None:
        print(f"Main Cluster: [{CLB:.3f}, {CUB:.3f}]")
    
    # 3. Access Detailed Results
    results = cluster.results()
    print(f"Cluster Width: {results['cluster_width']}")
    
    # 4. Custom Visualization
    cluster.plot(figsize=(15, 10))
    ```

=== "Output"

    ![Data Cluster](image/data_cluster/1770038710150.png)

---

## Notes

- **Unified Method**: Clustering now uses a single consistent method (normalized derivative threshold) for all GDF types.
- **Normalization**: All PDFs are normalized to [0, 1] to ensure the `derivative_threshold` behaves consistently regardless of the original data scale.
- **Requirements**: The input `gdf` must be fitted with `catch=True` so that `pdf_points` are available for analysis.

---

**Author:** Nirmal Parmar   
**Date:** 2025-10-10
