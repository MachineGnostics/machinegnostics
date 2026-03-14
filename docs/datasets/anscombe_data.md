# make_anscombe_check_data: Anscombe's Quartet Dataset

The `make_anscombe_check_data` function retrieves one of the four datasets from Anscombe's Quartet. These datasets are famous for being statistically identical (same mean, variance, correlation, and regression line) yet visually distinct, highlighting the importance of graphing data and using robust methods.

---

## Overview

Anscombe's quartet comprises four datasets that demonstrate the pitfalls of relying solely on simple descriptive statistics. They are ideal for benchmarking robust regression and gnostic models against standard least-squares approaches.

-   **Dataset 1**: A simple linear relationship with Gaussian noise. This is the "standard" case where simple linear regression works well.
-   **Dataset 2**: A clear non-linear (polynomial) relationship. Linear regression fails to capture the curve.
-   **Dataset 3**: A tight linear relationship with a single severe outlier. The outlier pulls the standard regression line significantly.
-   **Dataset 4**: A vertical line (constant X) with one influential outlier point. This point solely determines the correlation and regression slope.

All four datasets share nearly identical:

-   Mean of X: 9.0
-   Mean of y: 7.50
-   Sample Variance of X: 11.0
-   Sample Variance of y: 4.12
-   Correlation between X and y: 0.816
-   Linear Regression Line: $y = 3.00 + 0.500x$

---

## Parameters

| Parameter    | Type | Description                                                                 | Default |
| :----------- | :--- | :-------------------------------------------------------------------------- | :------ |
| `dataset_id` | int  | The identifier of the dataset to retrieve (1, 2, 3, or 4).                  | `1`     |

---

## Returns

| Return | Type          | Description                           |
| :----- | :------------ | :------------------------------------ |
| `X`    | numpy.ndarray | The input feature array of shape `(11,)`. |
| `y`    | numpy.ndarray | The target array of shape `(11,)`.    |

---

## Raises

-   **ValueError**
    If `dataset_id` is not 1, 2, 3, or 4.

---

## Example Usage

```python
from machinegnostics.data import make_anscombe_check_data
import numpy as np

# Load Dataset 3 (Linear with outlier)
X, y = make_anscombe_check_data(dataset_id=3)

print(f"Dataset 3 - X mean: {np.mean(X):.2f}, y mean: {np.mean(y):.2f}")
# Output: X mean: 9.00, y mean: 7.50
```

---

**Source:** Anscombe, F. J. (1973). "Graphs in Statistical Analysis". American Statistician. 27 (1): 17–21.

**Author:** Nirmal Parmar
