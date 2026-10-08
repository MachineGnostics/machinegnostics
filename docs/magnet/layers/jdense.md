# jDense — Gnostic Quantifying Weight Dense Layer

Dense layer that applies gnostic quantifying weights (`gnostic_weights_j`) before the affine transformation.

---

## Overview

`jDense` mirrors `iDense` but uses the quantifying weight family (`gnostic_weights_j`) — computed as the inverse of the estimating weights — before the dense affine transform. It is the quantification-side counterpart to `iDense`, emphasizing the `fj`-derived weighting path.

---

## Key Features

- Applies `gnostic_weights_j()` (inverse of estimating weights) before dense transform
- Configurable scale `S` and `lel`
- Full inheritance from `Dense`

---

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `in_features` | `int` | Required | Input dimension |
| `out_features` | `int` | Required | Output dimension |
| `S` | `float` \| `str` | `2.0` | Scale parameter |
| `lel` | `float` | `None` | Local Estimate of Location |

---

## Methods

### `forward(x, training=True)`

Compute quantifying weights from `x`, multiply into input, then apply dense affine transform.

---

## Example Usage

```python
from machinegnostics.magnet import jDense
layer = jDense(in_features=2, out_features=1)
print(layer(np.array([[1.0, 2.0]])).shape)
```

---

**Author:** Nirmal Parmar
