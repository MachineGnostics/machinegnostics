# Softmax — Normalized Probability Activation

Converts logits to normalized probabilities across the last axis. Subtracts max logit before exponentiation for numerical stability.

---

## Overview

Used in multiclass classification heads. Implementation subtracts maximum along last axis, then exponentiates and normalizes to sum to 1.

---

## Key Features

- Stable numerical computation (max subtraction)
- Probability output sums to 1 across final axis
- Works with batch dimensions

---

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
`name` | `str \| None` | `None` | Layer name
`verbose` | `bool` | `False` | Debug logging

---

## Methods

### `forward(x, training=True)`

Return the activation applied to input.

- **x**: `Tensor` or array-like — Input values of any shape
- **Returns**: `Tensor` with same shape as `x`

---

## Example Usage

```python
import numpy as np
from machinegnostics.magnet import Softmax

activation = Softmax()
result = activation(np.array([[[1.0, 2.0, 3.0]]]))
print(result.data.tolist())  # [[[0.09, 0.24, 0.67]]]
```

---

**Author:** Nirmal Parmar
