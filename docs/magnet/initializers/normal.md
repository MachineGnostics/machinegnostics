# Normal — Random Normal Weight Initializer

Samples weights from a normal (Gaussian) distribution with configurable mean and standard deviation.

---

## Overview

`Normal` draws weight values from a normal distribution centered at `mean` with standard deviation `stddev`. It provides a flexible, distribution-based initialization strategy for neural network weights. Reproducible via `seed`.

---

## Key Features

- Configurable mean and standard deviation
- Reproducible via `seed`
- Returns `np.float64` array by default
- Alias: `RandomNormal`

---

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `mean` | `float` | `0.0` | Mean of the normal distribution |
| `stddev` | `float` | `0.05` | Standard deviation of the normal distribution |
| `seed` | `int` \| `None` | `None` | Random seed for reproducibility |
| `dtype` | `np.dtype` | `np.float64` | Output data type |

---

## Methods

### `__call__(shape)`

Return normal random samples with the requested shape.

---

## Example Usage

```python
from machinegnostics.magnet import Normal
init = Normal(mean=0.0, stddev=0.05, seed=42)
W = init((10, 20))
print(W.shape)  # (10, 20)
```

---

**Author:** Nirmal Parmar
