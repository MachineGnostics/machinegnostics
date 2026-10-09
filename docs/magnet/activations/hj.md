# Hj — Gnostic Irrelevance Activation

Gnostic irrelevance / complement activation measuring deviation from concept center `z0` via `tanh(2θ)` with learnable scale `S`. Value range `[-∞, ∞]` . 

---

## Overview

Works alongside `Hi` for paired relevance-irrelevance characterization of feature alignment.

The centered coordinate is `θ = (x - z0) / S`.

---

## Key Features

- Learnable `z0`, `S`
- Irrelevance / deviation measure
- Complement to `Hi`

---

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
`learnable_S` | `bool` | `True` | Optimize scale
`learnable_z0` | `bool` | `True` | Optimize center
`initial_S` | `float` | `1.0` | Initial scale
`initial_z0` | `float \| None` | `None` | Initial center
`name` | `str \| None` | `None` | Name
`verbose` | `bool` | `False` | Logging

---

## Attributes

- `S`: Stored raw scale parameter tensor
- `z0`: Learnable center tensor
- `theta`: Most recent centered deviation (`θ`)
- `last_output`: Most recent irrelevance response

---

## Methods

### `forward(x, training=True)`

Return the activation applied to input.

- **x**: `Tensor` or array-like — Input values of any shape
- **Returns**: `Tensor` with same shape as `x`

---

## Example Usage

```python
import numpy as np
from machinegnostics.magnet import Hj

activation = Hj()
result = activation(np.array([[0.0]]))
print(result.data.tolist())  # [[0.0]]
```

---

**Author:** Nirmal Parmar
