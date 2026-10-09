# Softplus — Smooth Positive Activation

Smooth approximation to ReLU: `log(1 + exp(x))`. Strictly positive, fully differentiable, no hard corner at zero.

---

## Overview

Useful when the model benefits from strictly positive outputs without introducing a discontinuity.

---

## Key Features

- Smooth positive output
- No hard corner at zero
- Fully differentiable everywhere

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
from machinegnostics.magnet import Softplus

activation = Softplus()
result = activation(np.array([[0.0]]))
print(result.data.tolist())  # [[0.693]]
```

---

**Author:** Nirmal Parmar
