# ISSLoss — Irrelevance Squared Sum Loss

Measures irrelevance (irrelevance squared sum) of the prediction residual relative to the gnostic concept center — how far predictions deviate from the learned manifold.

---

## Overview

The conceptual form is `L = mean(hj²)` with `hj = sinh(2θ)` and `θ = (y_pred - y_true) / S`. This numerically stable surrogate avoids singularities while emphasizing large directional residuals. The centered coordinate is `θ = (y_pred - y_true) / S`.

Value range: `[0, ∞)`.

---

## Key Features

- Surrogate for `mean(hj²)`
- Custom autograd kernel (`ISSLossFunction`)
- Strong gradient for off-center predictions

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

Return batch inverse-style residual surrogate (`mean(sinh²(2θ))`).

---

## Example Usage

```python
from machinegnostics.magnet import ISSLoss, Tensor
import numpy as np
y_pred = Tensor(np.zeros((2, 1)), requires_grad=True)
y_true = Tensor(np.zeros((2, 1)))
print(float(ISSLoss()(y_pred, y_true)))  # 0.0
```

---

**Author:** Nirmal Parmar  
