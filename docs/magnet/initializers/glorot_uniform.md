# GlorotUniform — Glorot Uniform Weight Initializer

Samples weights from a uniform distribution scaled by `sqrt(6 / (fan_in + fan_out))`.

---

## Overview

`GlorotUniform` (Xavier Uniform) balances forward and backward variance over input and output dimensions. Scaled to `sqrt(6 / (fan_in + fan_out))` for uniform sampling, ideal for networks using tanh/sigmoid activations.

---

## Key Features

- Balanced uniform scaling: `sqrt(6 / (fan_in + fan_out))`
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

Return Glorot-scaled uniform samples with validated dimensions.

---

## Example Usage

```python
from machinegnostics.magnet import GlorotUniform
init = GlorotUniform(seed=42)
W = init((10, 20))
print(W.shape)  # (10, 20)
```

---

**Author:** Nirmal Parmar
