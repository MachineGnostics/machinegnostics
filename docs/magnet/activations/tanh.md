# Tanh — Hyperbolic Tangent Activation

Squashes inputs to `(-1, 1)` with zero-centered output. Useful when subsequent layers benefit from positive and negative responses of similar scale.

---

## Overview

Hyperbolic tangent is smooth, zero-centered, and bounded. It avoids the positive-bias of ReLU and can help with gradient symmetry in deep stacks.

---

## Key Features

- Zero-centered output in `(-1, 1)`
- Smooth, fully differentiable
- Avoids positive-only bias of ReLU

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
from machinegnostics.magnet import Tanh

activation = Tanh()
result = activation(np.array([[0.0]]))
print(result.data.tolist())  # [[0.0]]
```

---

**Author:** Nirmal Parmar
