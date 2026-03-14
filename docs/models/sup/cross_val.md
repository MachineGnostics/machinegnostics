# CrossValidator: Custom k-Fold Cross-Validation

The `CrossValidator` class provides a simple, flexible implementation of k-fold cross-validation for evaluating machine learning models. It is designed to work with any model that implements `fit(X, y)` and `predict(X)` methods, and supports custom scoring functions for regression or classification tasks.

---

## Overview

Cross-validation is a robust technique for assessing the generalization performance of machine learning models. The `CrossValidator` class splits your dataset into `k` folds, trains the model on `k-1` folds, and evaluates it on the remaining fold, repeating this process for each fold. The results are aggregated to provide a reliable estimate of model performance.

---

## Parameters

| Parameter       | Type       | Default | Description                                                             |
| --------------- | ---------- | ------- | ----------------------------------------------------------------------- |
| `model`       | object     | —      | A machine learning model with `fit(X, y)` and `predict(X)` methods. |
| `X`           | array-like | —      | Feature matrix of shape `(n_samples, n_features)`. Accepts NumPy arrays or Pandas DataFrame.                    |
| `y`           | array-like | —      | Target labels of shape `(n_samples,)`. Accepts NumPy arrays or Pandas Series/DataFrame column.                                |
| `k`           | int        | 5       | Number of folds for cross-validation.                                   |
| `shuffle`     | bool       | True    | Whether to shuffle the dataset before splitting into folds.             |
| `random_seed` | int/None   | None    | Seed for reproducible shuffling (ignored if `shuffle=False`).         |

---

## Attributes

- **folds**: `list of tuple`
  List of `(train_indices, test_indices)` for each fold.

---

## Methods

### `split()`

Splits the dataset into `k` folds.

- **Returns**:
  `folds`: list of tuple
  Each tuple contains `(train_indices, test_indices)` for a fold.

### `evaluate(scoring_func)`

Performs k-fold cross-validation and returns evaluation scores.

- **Parameters**:`scoring_func`: callableA function that takes `y_true` and `y_pred` and returns a numeric score (e.g., `mean_squared_error`, `accuracy_score`).
- **Returns**:
  `scores`: list of float
  Evaluation scores for each fold.

---

## Example Usage

```python
from machinegnostics.models import CrossValidator, LinearRegressor
from machinegnostics.metircs import mean_squared_error
import numpy as np

# Generate random data
X = np.random.rand(100, 10)
y = np.random.rand(100)

# Initialize model and cross-validator
model = LinearRegressor()
cv = CrossValidator(model, X, y, k=5, shuffle=True, random_seed=42)

# Evaluate using mean squared error
scores = cv.evaluate(mean_squared_error)
print("Cross-Validation Scores:", scores)
print("Mean Score:", np.mean(scores))
```

---

## Notes

- The model is re-initialized and trained from scratch for each fold.
- Supports any model with `fit` and `predict` methods.
- Works with any scoring function that accepts `y_true` and `y_pred`.
- Shuffling with a fixed `random_seed` ensures reproducible splits.

---

**Author:** Nirmal Parmar  
**Date:** 2025-05-01

---
