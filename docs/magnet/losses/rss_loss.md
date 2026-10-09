# RSSLoss — Relevance Squared Sum Loss

Relevance-based objective measuring how much of the centered residual is relevant within the gnostic concept space.

---

## Overview

Uses bounded surrogate `mean(hi²)` where `hi = tanh(2θ)` with `θ = (y_pred - y_true) / S`. Behaves like a conventional regression objective (RSS) while remaining aligned with MAGNET's centered-coordinate framework. The centered coordinate is `θ = (y_pred - y_true) / S`.

Value range: `[0, ∞)` (bounded by tanh saturation; zero at perfect alignment).

---

## Key Features

- Bounded RSS surrogate (`mean(tanh²(2θ))`)
- Custom autograd kernel (`RSSLossFunction`)
- Compatible with standard regression workflows

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

Return batch residual-squared surrogate (`mean(tanh²(2θ))`).

---

## Example Usage

```python
from machinegnostics.magnet import RSSLoss, Tensor
import numpy as np
y_pred = Tensor(np.zeros((2, 1)), requires_grad=True)
y_true = Tensor(np.zeros((2, 1)))
print(float(RSSLoss()(y_pred, y_true)))  # 0.0
```

---

**Author:** Nirmal Parmar  
