# make_moons_check_data: Synthetic Two Moons Dataset

The `make_moons_check_data` function generates a synthetic "two moons" dataset. This classic dataset consists of two interleaving half-circles and is widely used for visualizing and benchmarking clustering and classification algorithms, particularly for testing non-linear decision boundaries.

---

## Overview

The dataset forms two crescent shapes that are not linearly separable. 

-   **Class 0 (Upper Moon)**: A half-circle arching upwards.
-   **Class 1 (Lower Moon)**: A half-circle arching downwards, shifted and interlocked with the upper moon.
-   **Purpose**: Ideal for testing kernel methods, neural networks, or advanced clustering algorithms (like `GnosticLocalClustering`) that can handle non-convex shapes.

---

## Parameters

| Parameter   | Type           | Description                                                        | Default |
| :---------- | :------------- | :----------------------------------------------------------------- | :------ |
| `n_samples` | int            | Total number of data points to generate.                           | `30`    |
| `noise`     | float or None  | Standard deviation of Gaussian noise added to data. None = No noise.| `None`  |
| `seed`      | int            | Random seed for reproducibility.                                   | `42`    |

---

## Returns

| Return | Type          | Description                                         |
| :----- | :------------ | :-------------------------------------------------- |
| `X`    | numpy.ndarray | Input feature array of shape `(n_samples, 2)`.      |
| `y`    | numpy.ndarray | Target label array of shape `(n_samples,)`.          |

---

## Example Usage

```python
from machinegnostics.data import make_moons_check_data
import numpy as np

# Generate noisy moon data
X, y = make_moons_check_data(n_samples=100, noise=0.1)

print(f"X shape: {X.shape}")
print(f"Unique classes: {np.unique(y)}")
# Output:
# X shape: (100, 2)
# Unique classes: [0 1]
```

---

**Author:** Nirmal Parmar
