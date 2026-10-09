# HeNormal — He Normal Weight Initializer

Samples weights from a normal distribution scaled by fan-in (`√(2/fan_in)`).

---

## Overview

`HeNormal` implements He initialization, designed for networks using ReLU-family activations. Scaling stddev by `√(2 / fan_in)` preserves gradient variance during backpropagation through ReLU units, reducing vanishing/exploding gradient risk.

---

## Key Features

- Scaled stddev by `√(2 / fan_in)` (He scaling)
- Suitable for ReLU-family activations
- Reproducible via `seed`
- Returns `np.float64` array by default

---

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `seed` | `int` \| `None` | `None` | Random seed for reproducibility |
| `dtype` | `np.dtype` | `np.float64` | Output data type |

---

## Methods

### `__call__(shape)`

Return He-scaled normal samples with validated dimensions.

---

## Example Usage

```python
from machinegnostics.magnet import HeNormal
init = HeNormal(seed=42)
W = init((10, 20))
print(W.shape)  # (10, 20)
```

---

**Author:** Nirmal Parmar
