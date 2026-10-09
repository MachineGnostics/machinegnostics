# Zeros — Constant Zeros Weight Initializer

Fills a weight tensor with constant zeros.

---

## Overview

`Zeros` returns an array of the requested shape filled entirely with `0.0`. It is the standard initialization for bias terms and is a common choice for parameters that should start from a neutral, zero state before learning begins.

---

## Key Features

- Constant value of `0.0` across the entire tensor
- Standard choice for bias initialization
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

Return an array of zeros with the requested shape and validated dimensions.

---

## Example Usage

```python
from machinegnostics.magnet import Zeros
init = Zeros()
b = init((20,))
print(b.shape)  # (20,)
print(b[:3])    # [0. 0. 0.]
```

---

**Author:** Nirmal Parmar
