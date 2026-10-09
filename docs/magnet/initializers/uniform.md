# Uniform — Uniform Random Weight Initializer

Samples weights from a uniform distribution over `[-limit, +limit]`, where `limit` is derived from the input dimension.

---

## Overview

`Uniform` draws weight values from a uniform distribution centered at zero, scaled by `sqrt(1 / fan_in)`. It provides a simple, distribution-free initialization strategy that works reasonably well across a variety of activation functions. Reproducible via `seed`.

---

## Key Features

- Uniform sampling over `[-limit, +limit]` with `limit = sqrt(1 / fan_in)`
- Works as a general-purpose default initializer
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

Return uniform random samples scaled by the fan-in of the target shape.

---

## Example Usage

```python
from machinegnostics.magnet import Uniform
init = Uniform(seed=42)
W = init((10, 20))
print(W.shape)  # (10, 20)
print(W.min(), W.max())  # bounds around 0
```

---

**Author:** Nirmal Parmar
