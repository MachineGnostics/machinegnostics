# HeUniform — He Uniform Weight Initializer

Samples weights from a uniform distribution scaled by `√(6/fan_in)`.

---

## Overview

`HeUniform` implements He initialization for uniform sampling. Like `HeNormal`, it scales by fan-in (`√(2 / fan_in)`) to preserve gradient variance, but uses a uniform distribution over `[-limit, +limit]` instead. Ideal for ReLU-family activation networks.

---

## Key Features

- Uniform scaling `√(6 / fan_in)`
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

Return He-scaled uniform samples with validated dimensions.

---

## Example Usage

```python
from machinegnostics.magnet import HeUniform
init = HeUniform(seed=42)
W = init((10, 20))
print(W.shape)  # (10, 20)
```

---

**Author:** Nirmal Parmar
