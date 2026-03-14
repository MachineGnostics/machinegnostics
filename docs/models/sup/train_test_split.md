# train_test_split: Random Train/Test Data Splitter

The `train_test_split` function provides a simple and flexible way to split your dataset into random training and testing subsets. It accepts NumPy arrays, Pandas DataFrame/Series, and can also handle lists or tuples as input. This function is essential for evaluating machine learning models on unseen data and is a core utility in most ML workflows.

---

## Overview

Splitting your data into training and testing sets is a fundamental step in machine learning. The `train_test_split` function allows you to:

- Randomly partition your data into train and test sets.
- Specify the proportion or absolute number of test samples.
- Shuffle your data for unbiased splitting.
- Use a random seed for reproducibility.
- Split both features (`X`) and targets (`y`) in a consistent manner.

---

## Parameters

| Parameter     | Type                | Default | Description                                                                 |
|---------------|---------------------|---------|-----------------------------------------------------------------------------|
| `X`           | array-like          | —       | Feature data to be split. Accepts NumPy arrays or Pandas DataFrame. Must be indexable and of consistent length.       |
| `y`           | array-like or None  | None    | Target data to be split alongside X. Accepts NumPy arrays or Pandas Series/DataFrame column. Must be same length as X.              |
| `test_size`   | float or int        | 0.25    | If float, fraction of data for test set (0.0 < test_size < 1.0). If int, absolute number of test samples. |
| `shuffle`     | bool                | True    | Whether to shuffle the data before splitting.                               |
| `random_seed` | int or None         | None    | Controls the shuffling for reproducibility.                                 |

---

## Returns

- **X_train, X_test**: `np.ndarray`  
  Train-test split of X.

- **y_train, y_test**: `np.ndarray` or `None`  
  Train-test split of y. If y is None, these will also be None.

---

## Raises

- **ValueError**  
  If inputs are invalid or `test_size` is not appropriate.

- **TypeError**  
  If `test_size` is not a float or int.

---

## Example Usage

```python
import numpy as np
from machinegnostics.models import train_test_split

# Create sample data
X = np.arange(20).reshape(10, 2)
y = np.arange(10)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=True, random_seed=42
)

print("X_train:", X_train)
print("X_test:", X_test)
print("y_train:", y_train)
print("y_test:", y_test)
```

---

## Notes

- If `y` is not provided, only `X` will be split and `y_train`, `y_test` will be `None`.
- If `test_size` is a float, it must be between 0.0 and 1.0 (exclusive).
- If `test_size` is an int, it must be between 1 and `len(X) - 1`.
- Setting `shuffle=False` will split the data in order, without randomization.
- Use `random_seed` for reproducible splits.

---

**Author:** Nirmal Parmar  
**Date:** 2025-05-01

---