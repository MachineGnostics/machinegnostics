# ReLU — Rectified Linear Unit Activation

The `ReLU` activation applies the rectified linear unit (`max(0, x)`) elementwise.

---

## Overview

ReLU is the default hidden-layer activation in many feed-forward networks because it is simple, inexpensive, and avoids heavy saturation behavior. It returns zero for negative inputs and preserves positive values unchanged.

---

## Key Features

- Simple, fast elementwise operation
- No saturation for positive inputs (gradients flow freely)
- Compatible with MAGNET `Tensor` facade
- Standard drop-in replacement for conventional networks

---

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `name` | `str` \| `None` | `None` | Optional layer name |
| `verbose` | `bool` | `False` | Enable debug logging |

---

## Methods

### `forward(x, training=True)`

Return `max(0, x)` elementwise.

- **x**: `Tensor` or array-like — Input values of any shape
- **Returns**: `Tensor` with same shape as `x`

---

## Example Usage

```python
import numpy as np
from machinegnostics.magnet import ReLU

activation = ReLU()
result = activation(np.array([-2.0, 0.0, 3.0]))
print(result.data.tolist())  # [0.0, 0.0, 3.0]
```

---

**Author:** Nirmal Parmar
