# Sigmoid — Logistic Activation

Maps each value into `(0, 1)` via `sigmoid(x)`. Common for binary output layers, probabilities, and gates.

---

## Overview

Standard smooth bounded nonlinearity. Subtracts max for numerical stability in Softmax; Sigmoid itself uses native torch sigmoid.

---

## Key Features

- Smooth, differentiable everywhere
- Maps to open interval `(0, 1)`
- Zero-centered gradient near 0.5

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
from machinegnostics.magnet import Sigmoid

activation = Sigmoid()
result = activation(np.array([[0.0]]))
print(result.data.tolist())  # [[0.5]]
```

---

**Author:** Nirmal Parmar
