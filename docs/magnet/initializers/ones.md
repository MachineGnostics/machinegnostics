# Ones — Constant Ones Weight Initializer

Fills a weight tensor with constant ones.

---

## Overview

`Ones` returns an array of the requested shape filled entirely with `1.0`. It is most commonly used to initialize bias-like terms or gating variables that should start at a non-zero unit value (for example, the scale parameter `s` in gnostic neurons).

---

## Key Features

- Constant value of `1.0` across the entire tensor
- Useful for bias/gate/scale initialization
- Shape validated against the initializer helpers
- Returns `np.float64` array by default

---

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `dtype` | `np.dtype` | `np.float64` | Output data type |

---

## Methods

### `__call__(shape)`

Return an array of ones with the requested shape and validated dimensions.

---

## Example Usage

```python
from machinegnostics.magnet import Ones
init = Ones()
b = init((20,))
print(b.shape)  # (20,)
print(b[:3])    # [1. 1. 1.]
```

---

**Author:** Nirmal Parmar
