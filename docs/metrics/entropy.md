# entropy: Gnostic Entropy Metric

The `entropy` function computes the Gnostic entropy of a data distribution or the entropy of the difference between two data samples. This metric evaluates uncertainty or disorder within the framework of Mathematical Gnostics, providing robust, assumption-free measurements.

---

## Overview

Gnostic entropy is a measure of the uncertainty associated with a dataset. It leverages the estimating (EGDF) or quantifying (QGDF) global distribution functions to determine the level of disorder:

- **Single Dataset**: Calculates the entropy of the distribution of `data`.
- **Two Datasets**: Calculates the entropy of the residuals (`data_compare - data`), useful for evaluating model errors.

The calculation depends on the selected geometry:

- **Case `'i'` (Estimation)**: `Entropy = 1 - mean(fi)`. Represents standard uncertainty, typically in [0, 1].
- **Case `'j'` (Quantification)**: `Entropy = mean(fj) - 1`. Used for quantifying outliers or extreme deviations.

---

## Parameters

| Parameter      | Type                                   | Description                                                                 |
|----------------|----------------------------------------|-----------------------------------------------------------------------------|
| `data`         | array-like or Pandas Series            | Reference data values (e.g., Ground Truth) or single dataset. Must be 1D.   |
| `data_compare` | array-like or Pandas Series, optional  | Data to compare (e.g., Predicted). Comparison is `data_compare - data`.     |
| `S`            | float or 'auto'          | Scale parameter. If float, suggested [0.01, 2]. Default: `'auto'`.          |
| `case`         | str                      | `'i'` for estimating (EGDF), `'j'` for quantifying (QGDF). Default: `'i'`.  |
| `z0_optimize`  | bool                     | Whether to optimize the location parameter z0. Default: `False`.            |
| `data_form`    | str                      | `'a'` for additive (diff), `'m'` for multiplicative. Default: `'a'`.        |
| `tolerance`    | float                    | Convergence tolerance for optimization. Default: `1e-6`.                    |
| `verbose`      | bool                     | If True, enables detailed logging. Default: `False`.                        |

---

## Returns

- **float**  
  The calculated Gnostic entropy value.

---

## Raises

- **TypeError**  
  If inputs are not array-like or have incorrect types.
- **ValueError**  
  If inputs have mismatched shapes, are empty, contain NaN/Inf, or if invalid options are provided.

---

## Example Usage

```python
import numpy as np
from machinegnostics.metrics import entropy

# Example 1: Entropy of a single dataset
data = np.random.normal(0, 1, 100)
ent = entropy(data, case='i')
print(f"Entropy (single): {ent}")

# Example 2: Entropy of residuals (Model Evaluation)
y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
ent_diff = entropy(data=y_true, data_compare=y_pred, case='i')
print(f"Entropy (residuals): {ent_diff}")

# Example 3: detecting outliers with case 'j'
y_outliers = np.array([1, 2, 3, 100])
ent_out = entropy(y_outliers, case='j')
print(f"Entropy (quantifying): {ent_out}")
```

---

## Notes

- For standard uncertainty estimation, use **case 'i'**. The values are typically normalized between 0 (certainty) and 1 (max uncertainty).
- For analyzing tails and outliers, use **case 'j'**.
- If `S='auto'`, the scale parameter is estimated automatically based on data homogeneity.
- Compatibility: Accepts NumPy arrays and Pandas Series.

---

**Author:** Nirmal Parmar  
**Date:** 2026-02-02
