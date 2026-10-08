# InformationLoss — Gnostic Information Loss

Measures information content of prediction residuals relative to the target-centered concept manifold.

---

## Overview

Quantifies how much information remains in residuals after mapping into MAGNET's gnostic terms. Uses a stabilized binary entropy proxy derived from `hi = tanh(2θ)` with `θ = (y_pred - y_true) / S`. The centered coordinate is `θ = (y_pred - y_true) / S`.

Value range: `[0, ∞)` (entropy-style, bounded by clamping).

---

## Key Features

- Stabilized binary entropy proxy
- Custom autograd kernel (`InformationLossFunction`)
- Information-theoretic interpretation

---

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `S` | `float` \| `str` | `1` | Scale parameter; `"auto"` for batch-local scale |
| `name` | `str` \| `None` | `None` | Optional display name |
| `verbose` | `bool` | `False` | Verbose logging |

---

## Attributes

- `S`: Scale parameter
- `S_local`: Resolved local scale

---

## Methods

### `forward(y_pred, y_true)`

Return batch-mean information objective.

---

## Example Usage

```python
from machinegnostics.magnet import InformationLoss, Tensor
import numpy as np
y_pred = Tensor(np.array([[0.1], [0.2]]), requires_grad=True)
y_true = Tensor(np.zeros((2, 1)))
print(np.isfinite(float(InformationLoss()(y_pred, y_true))))  # True
```

---

**Author:** Nirmal Parmar  
