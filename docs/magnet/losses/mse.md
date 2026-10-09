# MSE — Mean Squared Error Loss

The `MSE` loss computes the mean squared error between predictions and targets for regression tasks.

---

## Overview

Standard regression objective computing `mean((y_pred - y_true)²)`. No gnostic scale `S` or centered coordinate `θ` is used; it applies direct squared difference.

Value range: `[0, ∞)`.

---

## Key Features

- Simple quadratic penalty
- Works with `Tensor` and array-like inputs
- Shape validation via `prepare_tensors()`
- Compatible with MAGNET training loops

---

## Methods

### `forward(y_pred, y_true)`

Return average squared difference.

- **y_pred**: `Tensor` or array-like — Predicted values
- **y_true**: `Tensor` or array-like — Target values (same shape)
- **Returns**: Scalar `Tensor`
- **Raises**: `ValueError` if shapes differ

---

## Example Usage

```python
import numpy as np
from machinegnostics.magnet import MSE

loss = MSE()
value = loss(np.array([[2.0]]), np.array([[1.0]]))
print(float(value))  # 1.0
```

---

**Author:** Nirmal Parmar  
