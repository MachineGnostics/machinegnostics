# silhouette_score: Silhouette Score Metric

The `silhouette_score` function computes the mean Silhouette Coefficient for a dataset given its cluster labels. It evaluates the quality of clustering by measuring how similar an object is to its own cluster (cohesion) compared to other clusters (separation).

---

## Overview

The Silhouette Coefficient is calculated for each sample using:

- **a**: The mean distance between a sample and all other points in the same cluster.
- **b**: The mean distance between a sample and all other points in the *nearest* cluster.

The coefficient for a single sample is given by:
\[ s = \frac{b - a}{\max(a, b)} \]

Expected values range from -1 to 1:

- **+1**: Indicates that the sample is far away from the neighboring clusters.
- **0**: Indicates that the sample is on or very close to the decision boundary between two neighboring clusters.
- **-1**: Indicates that those samples might have been assigned to the wrong cluster.

---

## Parameters

| Parameter | Type                               | Description                                              |
| --------- | ---------------------------------- | -------------------------------------------------------- |
| `X`       | array-like or Pandas DataFrame     | Feature data of shape (n_samples, n_features).           |
| `labels`  | array-like or Pandas Series        | Cluster labels for each sample. Shape (n_samples,).      |
| `verbose` | bool                               | If True, enables detailed logging for debugging. Default: `False`. |

---

## Returns

- **float**  
  The mean Silhouette Coefficient for all samples.

---

## Raises

- **TypeError**  
  If inputs `X` or `labels` are not array-like.
- **ValueError**  
  If dimensions are incorrect, data is empty, contains NaN/Inf, or if number of labels doesn't match samples.

---

## Example Usage

```python
from machinegnostics.metrics import silhouette_score
import numpy as np

# Example: Clustering assessment
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])
labels = np.array([0, 0, 0, 1, 1, 1])

score = silhouette_score(X, labels)
print(f"Silhouette Score: {score}")
```

---

## Notes

- Compatibility: Accepts NumPy arrays and Pandas DataFrame/Series.
- Requires at least 2 distinct clusters to be calculated.
- If the number of unique labels is 1 or equal to the number of samples, the score is 0.0.
- Uses Euclidean distance for calculations.
- This metric is useful for selecting the optimal number of clusters (e.g., in K-Means).

---

**Author:** Nirmal Parmar  
**Date:** 2026-02-02
