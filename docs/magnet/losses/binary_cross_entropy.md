# BinaryCrossEntropy — Binary Classification Loss

Compute binary cross-entropy for probability predictions.

---

## Overview

Appropriate when model outputs Bernoulli probabilities, usually after sigmoid. Penalizes overconfident incorrect predictions heavily while rewarding accurate, calibrated probabilities. Includes numerical stability floor (`eps=1e-12`).

Standard binary classification objective using `-mean(y_true · log(y_pred) + (1 - y_true) · log(1 - y_pred))`. No gnostic scale `S` or `θ` coordinate applies.

Value range: `[0, ∞)`.

---

## Key Features

- Standard binary classification objective
- Numerical stability with `eps` clipping
- MAGNET `Tensor` compatible

---

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `y_pred` | `Tensor` / array | — | Predicted probabilities in `[0, 1]` |
| `y_true` | `Tensor` / array | — | Binary targets (`0` / `1`) |
| `eps` | `float` | `1e-12` | Stability floor before log |

---

## Methods

### `forward(y_pred, y_true, eps=1e-12)`

Return mean binary cross-entropy.

- **Returns**: Scalar `Tensor`

---

## Example Usage

```python
import numpy as np
from machinegnostics.magnet import BinaryCrossEntropy
loss = BinaryCrossEntropy()
value = loss(np.array([[0.9]]), np.array([[1.0]]))
```

---

**Author:** Nirmal Parmar  
