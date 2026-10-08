# ELU — Exponential Linear Unit

Identity for positive inputs; smooth exponential for negative inputs (`alpha=1.0`). Reduces dead-unit behavior while keeping positive-region gradients intact.

---

## Overview

ELU transitions smoothly into an exponential curve for negative inputs, avoiding the hard corner at zero.

---

## Key Features

- Smooth transition for negatives
- Reduces dead-unit behavior
- Positive region identity

---

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
`alpha` | `float` | `1.0` | Saturation scale
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
from machinegnostics.magnet import ELU

activation = ELU()
result = activation(np.array([[-1.0, 1.0]]))
print(result.data.tolist())  # [[-0.63, 1.0]]
```

---

**Author:** Nirmal Parmar
