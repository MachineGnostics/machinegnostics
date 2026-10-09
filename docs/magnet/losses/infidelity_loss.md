# InfidelityLoss — Gnostic Infidelity Loss

Penalize inverse fidelity so off-concept predictions become expensive.

---

## Overview

`InfidelityLoss` minimizes the batch mean of the infidelity characteristic `fj = cosh(2θ)` with `θ = (y_pred - y_true) / S`. Because `fj` is bounded in `[1, ∞)`, the loss emphasizes poor concept alignment more aggressively than fidelity-only objectives. 

Value range: `[1, ∞)` (minimum `1` at perfect alignment).

---

## Key Features

- Custom autograd kernel (`InfidelityLossFunction`)
- Scale parameter `S` (numeric or `"auto"`)
- Sharp penalty for low-fidelity predictions

---

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `S` | `float` \| `str` | `1` | Scale parameter; `"auto"` for local batch scale |
| `name` | `str` \| `None` | `None` | Optional display name |
| `verbose` | `bool` | `False` | Verbose logging |

---

## Attributes

- `S`: Scale parameter
- `S_local`: Resolved local scale

---

## Methods

### `forward(y_pred, y_true)`

Return mean inverse-fidelity objective (`mean(cosh(2θ))`).

---

## Example Usage

```python
import numpy as np
from machinegnostics.magnet import InfidelityLoss, Tensor
y_pred = Tensor(np.zeros((2, 1)), requires_grad=True)
y_true = Tensor(np.zeros((2, 1)))
print(float(InfidelityLoss()(y_pred, y_true)))  # 1.0
```

---

**Author:** Nirmal Parmar  
