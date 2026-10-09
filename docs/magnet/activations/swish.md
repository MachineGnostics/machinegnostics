# Swish — Self-Gated Nonlinearity

Computes `x * sigmoid(x)`. Smooth self-gated function; often performs well as ReLU replacement.

---

## Overview

Preserves small negative responses while remaining fully differentiable. Uses sigmoid gating rather than fixed threshold.

---

## Key Features

- Self-gated (`sigmoid(x)` factor)
- Smooth, fully differentiable
- Often outperforms ReLU

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
from machinegnostics.magnet import Swish

activation = Swish()
result = activation(np.array([[0.0, 1.0]]))
print(result.data.tolist())  # [[0.0, 0.731]]
```

---

**Author:** Nirmal Parmar
