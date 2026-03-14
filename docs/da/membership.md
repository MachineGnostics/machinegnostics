# DataMembership: Gnostic Membership Test (Machine Gnostics)

The `DataMembership` class provides a robust method to test whether a value can be considered a member of a homogeneous data sample, using the EGDF (Estimating Global Distribution Function) framework. It determines the bounds within which new data points can be added to a sample without disrupting its homogeneity.

---

## Overview

**Membership Test:**
"Is a value Zξ a potential member of the given sample Z?" In other words: "Will the homogeneous sample Z remain homogeneous after extension by Zξ?"

**Logic Process:**

1.  **Homogeneity Check**: First, check if the sample Z is homogeneous using `DataHomogeneity`.
2.  **Extension Test**: If Z is homogeneous, extend the sample with a candidate value Zξ and check if the extended sample remains homogeneous.
3.  **Bound Search**: Determine the Lower Sample Bound (LSB) and Upper Sample Bound (USB).
    *   LSB search range: [LB, DLB]
    *   USB search range: [DUB, UB]
4.  **Result**: Find the minimum and maximum values of Zξ that preserve homogeneity.

---

## Key Features

- **Homogeneity Preservation**: Membership is defined by the preservation of sample homogeneity.
- **Adaptive Bound Search**: Iteratively finds the exact boundaries where homogeneity is lost.
- **Diagnostic Visualization**: Plots EGDF, PDF, and the determined membership bounds.

---

## Parameters

| Parameter             | Type    | Default | Description                                                  |
| --------------------- | ------- | ------- | ------------------------------------------------------------ |
| `gdf`                 | EGDF    | -       | Fitted EGDF object (must be fitted and contain data).        |
| `verbose`             | bool    | False   | If True, detailed logs are printed during execution.         |
| `catch`               | bool    | True    | If True, errors and warnings are stored in `params`.         |
| `tolerance`           | float   | 1e-3    | Tolerance level for numerical calculations.                  |
| `max_iterations`      | int     | 100     | Maximum number of iterations for bound search.               |
| `initial_step_factor` | float   | 0.001   | Initial step size factor for adaptive bound search.          |

---

## Attributes

- **LSB**: `float` or `None`
  The calculated Lower Sample Bound.
- **USB**: `float` or `None`
  The calculated Upper Sample Bound.
- **is_homogeneous**: `bool`
  Indicates whether the original data sample is homogeneous.
- **params**: `dict`
  Stores results, errors, warnings, and search parameters.
- **fitted**: `bool`
  Indicates whether the membership analysis has been completed.

---

## Methods

### `fit()`

Performs the membership analysis to determine the Lower Sample Bound (LSB) and Upper Sample Bound (USB).

**Returns:**
`Tuple[Optional[float], Optional[float]]` — The calculated LSB and USB values.

---

### `plot()`

Generates a plot of the EGDF and PDF with membership bounds.

**Parameters:**

- `plot_smooth` (bool, default=True): If True, plots smoothed curves.
- `plot` (str, default='both'): 'gdf', 'pdf', or 'both'.
- `bounds` (bool, default=True): If True, includes data bounds.
- `figsize` (tuple, default=(12, 8)): Figure size.

**Returns:**
None

---

### `results()`

Returns the analysis results stored in the `params` attribute.

**Returns:**
`Dict[str, Any]` — Analysis results including LSB, USB, homogeneity status, and parameters.

---

## Example Usage

=== "Python"

    ```python
    import numpy as np
    from machinegnostics.magcal import EGDF, DataMembership
    
    # 1. Prepare homogeneous data
    data = np.array([ -13.5, 0, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
    
    # 2. Fit EGDF
    egdf = EGDF(data=data, catch=True)
    egdf.fit()
    
    # 3. Perform Membership Analysis
    membership = DataMembership(gdf=egdf, verbose=True)
    lsb, usb = membership.fit()
    
    print(f"Lower Bound: {lsb}")
    print(f"Upper Bound: {usb}")
    
    # 4. Visualize Results
    membership.plot()
    
    # 5. Get detailed results
    results = membership.results()
    print(results)
    ```

=== "Output"

    ![Data Membership](image/membership/1770038667188.png)

---

## Notes

- **Requirement**: This analysis only works with **EGDF** objects.
- **Prerequisite**: The initial sample must be homogeneous. If `DataHomogeneity` finds it non-homogeneous, the analysis proceeds no further.
- **Output**: LSB and USB represent the safe range for extending the dataset while maintaining its structural distribution properties.

---

**Author:** Nirmal Parmar   
**Date:** 2025-10-10
