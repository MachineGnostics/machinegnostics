# ResidualEntropyLoss — Entropy of Residuals

Measures the entropy of residuals — the residual entropy of the data.

---

## Overview

The residual entropy is evaluated by `mean(fj - fi)` with `fj = cosh(2θ)` and `fi = sech(2θ)`, where the centered coordinate is `θ = (y_pred - y_true) / S`. This complements fidelity: high entropy indicates predictions are off-concept.

Value range: `[0, ∞)` (grows without bound as fidelity collapses).

---

## Key Features

- Residual entropy surrogate (`fj - fi`)
- Custom autograd kernel (`ResidualEntropyLossFunction`)
- Complements `FidelityLoss`

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

Return batch residual-entropy objective (`mean(cosh(2θ) - sech(2θ))`).

---

## Example Usage

```python
from machinegnostics.magnet import ResidualEntropyLoss, Tensor
import numpy as np
y_pred = Tensor(np.zeros((2, 1)), requires_grad=True)
y_true = Tensor(np.zeros((2, 1)))
print(float(ResidualEntropyLoss()(y_pred, y_true)))  # 0.0
```

---

**Author:** Nirmal Parmar  
