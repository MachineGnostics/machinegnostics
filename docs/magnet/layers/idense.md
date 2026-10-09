# iDense — Gnostic Estimating Weight Dense Layer

Dense layer that applies gnostic estimating weights (`gnostic_weights_i`) before the affine transformation.

---

## Overview

`iDense` extends `Dense` by multiplying the input by gnostic estimating weights computed from the current batch. This emphasizes the estimation-side weighting path (`fi`-based weights) before the standard `y = x @ W + b` affine transform. It inherits all `Dense` parameters and adds `S` (scale) and `lel` (local estimate of location) for the gnostic computation.

---

## Key Features

- Applies `gnostic_weights_i()` before dense transform
- Configurable scale `S` and `lel`
- Inherits full `Dense` parameter and method interface
- Supports `training` mode flag

---

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `in_features` | `int` | Required | Input dimension (`n_in`) |
| `out_features` | `int` | Required | Output dimension (`n_out`) |
| `S` | `float` \| `str` | `2.0` | Scale parameter for gnostic weight calculation |
| `lel` | `float` | `None` | Local Estimate of Location; computed as median if `None` |
| `weight_init` | `callable` | `XavierUniform` | Weight initializer |
| `bias_init` | `callable` | `Zeros()` | Bias initializer |

---

## Methods

### `forward(x, training=True)`

Compute estimating weights from `x`, reshape for broadcasting, multiply into input, then apply dense affine transform.

---

## Example Usage

```python
from machinegnostics.magnet import iDense
layer = iDense(in_features=2, out_features=1, S=2.0)
print(layer(np.array([[1.0, 2.0]])).shape)  # (1, 1)
```

---

**Author:** Nirmal Parmar
