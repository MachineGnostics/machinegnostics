# LeakyReLU — Leaky Rectified Linear Unit

Like ReLU but keeps small slope (default `alpha=0.01`) for negative inputs, allowing gradients to flow through inactive units.

---

## Overview

Prevents dead-unit behavior seen in plain ReLU by preserving a small negative gradient channel.

---

## Key Features

- Small negative slope (`alpha`)
- Prevents dead neurons
- Simple, fast elementwise

---

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
`alpha` | `float` | `0.01` | Negative slope
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
from machinegnostics.magnet import LeakyReLU

activation = LeakyReLU()
result = activation(np.array([[-2.0, 3.0]]))
print(result.data.tolist())  # [[-0.2, 3.0]]
```

---

**Author:** Nirmal Parmar
