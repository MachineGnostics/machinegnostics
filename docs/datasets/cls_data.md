# make_classification_check_data: Synthetic Classification Dataset

The `make_classification_check_data` function generates a synthetic dataset for validating classification models within the Machine Gnostics framework. This function creates simple blob-based clusters using Gaussian distributions, serving as a reliable "hello world" test for classification algorithms.

---

## Overview

This utility simplifies the creation of classification datasets by generating clusters of points centered around randomly positioned centroids.

-   **Method**: Gaussian blobs with configurable separability.
-   **Purpose**: Unit testing models, verifying pipeline integrity, and demonstrating basic classification capabilities.
-   **Customization**: Easily adjust the number of samples, features, classes, and task difficulty (separability).

---

## Parameters

| Parameter      | Type   | Description                                                                 | Default |
| :------------- | :----- | :-------------------------------------------------------------------------- | :------ |
| `n_samples`    | int    | Total number of data points to generate.                                    | `30`    |
| `n_features`   | int    | Number of input features (dimensions) per sample.                           | `2`     |
| `n_classes`    | int    | Number of distinct classes (labels).                                        | `2`     |
| `separability` | float  | Distance multiplier for class centers. Higher values = easier separation. | `2.0`   |
| `seed`         | int    | Random seed for reproducibility.                                            | `42`    |

---

## Returns

| Return | Type          | Description                                         |
| :----- | :------------ | :-------------------------------------------------- |
| `X`    | numpy.ndarray | Input feature array of shape `(n_samples, n_features)`. |
| `y`    | numpy.ndarray | Target label array of shape `(n_samples,)`.          |

---

## Example Usage

```python
from machinegnostics.data import make_classification_check_data
import numpy as np

# Generate a 3-class dataset with 50 samples
X, y = make_classification_check_data(n_samples=50, n_classes=3)

print(f"X shape: {X.shape}")
print(f"Unique classes: {np.unique(y)}")
# Output:
# X shape: (50, 2)
# Unique classes: [0 1 2]
```

---

**Author:** Nirmal Parmar
