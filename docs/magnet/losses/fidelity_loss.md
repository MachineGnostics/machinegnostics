# FidelityLoss — Gnostic Fidelity Loss

The `FidelityLoss` rewards predictions that align with the target-centered concept manifold by maximizing `mean(sech(2θ))`.

---

## Overview

For each residual-centered coordinate `θ = (y_pred - y_true) / S`, the loss evaluates the fidelity response `fi = sech(2θ)` and maximizes its mean. The objective lives in `[0, 1)`, reaches optimum `1` when predictions match targets exactly, and approaches `0` as fidelity collapses.

---

## Key Features

- Custom autograd kernel (`FidelityLossFunction`) for efficient gradients
- Scale parameter `S` may be numeric or `"auto"` (batch-derived)
- `y_true` supplies fixed concept center and never receives gradients
- Gradient clipping for numerical stability

---

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `S` | `float` \| `str` | `1` | Scale parameter; use `"auto"` for local batch scale |
| `name` | `str` \| `None` | `None` | Optional display name |
| `verbose` | `bool` | `False` | Verbose logging |

---

## Example Usage

```python
import numpy as np
from machinegnostics.magnet import FidelityLoss, Tensor

y_pred = Tensor(np.array([[0.0], [0.0]]), requires_grad=True)
y_true = Tensor(np.array([[0.0], [0.0]]))
print(float(FidelityLoss()(y_pred, y_true)))  # 1.0
```

---

**Author:** Nirmal Parmar  
