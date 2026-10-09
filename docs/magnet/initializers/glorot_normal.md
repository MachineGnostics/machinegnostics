# GlorotNormal — Glorot Normal Weight Initializer

Samples weights from a normal distribution scaled by `√(2/(fan_in + fan_out))`.

---

## Overview

`GlorotNormal` (also known as Xavier Normal) uses variance scaled by both input and output fan sizes to balance forward and backward signal propagation. Designed for activations like Tanh and Sigmoid that suffer from gradient attenuation.

---

## Key Features

- Balanced scaling: `√(2 / (fan_in + fan_out))`
- Suitable for tanh/sigmoid-style networks
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

Return Glorot-scaled normal samples with validated dimensions.

---

## Example Usage

```python
from machinegnostics.magnet import GlorotNormal
init = GlorotNormal(seed=42)
W = init((10, 20))
print(W.shape)  # (10, 20)
```

---

**Author:** Nirmal Parmar
