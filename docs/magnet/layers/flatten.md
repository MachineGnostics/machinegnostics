# Flatten — Reshape Layer

Reshape structured inputs into a 2D batch suitable for dense layers.

---

## Overview

`Flatten` collapses all dimensions after the batch axis into a single feature axis, producing output of shape `(batch, -1)`. It records the original `input_shape` for debugging. No trainable parameters are used.

---

## Key Features

- Collapses all non-batch dimensions
- No trainable parameters
- Preserves batch axis
- Records original `input_shape`

---

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `name` | `str` \| `None` | `None` | Layer name |
| `verbose` | `bool` | `False` | Debug logging |

---

## Methods

### `forward(x, training=True)`

Flatten input to `(batch, -1)`.

---

## Example Usage

```python
from machinegnostics.magnet import Flatten
layer = Flatten()
print(layer(np.ones((2, 3, 4))).shape)  # (2, 12)
```

---

**Author:** Nirmal Parmar
