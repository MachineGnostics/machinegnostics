# Square — Elementwise Square Activation

Returns `x²` elementwise. Emphasizes magnitude regardless of sign.

---

## Overview

Simple polynomial activation useful for symmetric response, energy-like objectives, or diagnostic experiments.

---

## Key Features

- Symmetric response (sign ignored)
- Simple polynomial
- Emphasizes magnitude

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
from machinegnostics.magnet import Square

activation = Square()
result = activation(np.array([[-2.0, 3.0]]))
print(result.data.tolist())  # [[4.0, 9.0]]
```

---

**Author:** Nirmal Parmar
