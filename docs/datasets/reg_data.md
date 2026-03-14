# make_regression_check_data: Synthetic Regression Dataset

The `make_regression_check_data` function generates synthetic regression datasets. It supports linear, polynomial, and sinusoidal relationships with additive Gaussian noise and optional outliers. This serves as a versatile testbed for validating regression models within the Machine Gnostics framework.

---

## Overview

This utility creates regression datasets for testing:

-   **Functions**: Linear/Polynomial ($y = \sum a_i x^i + c$), Sinusoidal ($y = A \sin(x) + c$), or Cosine.
-   **Outliers**: Option to contaminate a percentage of data with large shifts (`outlier_ratio`), ideal for testing robust regression capabilities.
-   **Purpose**: Unit testing models, verifying pipeline integrity, and demonstrating regression capabilities.
-   **Reproducibility**: Uses a fixed seed (default 42).

---

## Parameters

| Parameter       | Type   | Description                                                                 | Default |
| :-------------- | :----- | :-------------------------------------------------------------------------- | :------ |
| `n_samples`     | int    | The number of data points to generate.                                      | `20`    |
| `slope`         | float  | True coefficient (linear/poly) or amplitude (sin/cos).                      | `3.5`   |
| `intercept`     | float  | The true intercept (bias) of the underlying relationship.                   | `10.0`  |
| `noise_level`   | float  | Standard deviation of Gaussian noise added to target.                         | `2.0`   |
| `degree`        | int    | Degree of the polynomial. Used if `function_type='poly'`.                   | `1`     |
| `function_type` | str    | Component function type: `'poly'`, `'sin'`, `'cos'`.                        | `'poly'`|
| `outlier_ratio` | float  | Proportion of samples to contaminate with outliers (0.0 to 1.0).            | `0.0`   |
| `seed`          | int    | Random seed for reproducibility.                                            | `42`    |

---

## Returns

| Return | Type          | Description                                         |
| :----- | :------------ | :-------------------------------------------------- |
| `X`    | numpy.ndarray | Input feature array of shape `(n_samples,)`. Values uniform in [0, 10]. |
| `y`    | numpy.ndarray | Target array of shape `(n_samples,)`.                 |

---

## Example Usage

```python
from machinegnostics.data import make_regression_check_data
import numpy as np

# Example 1: Standard Linear Regression
X, y = make_regression_check_data(n_samples=50)

# Example 2: Sinusoidal data with outliers
X_sin, y_sin = make_regression_check_data(
    n_samples=100, 
    function_type='sin', 
    outlier_ratio=0.1
)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
```

---

**Author:** Nirmal Parmar
