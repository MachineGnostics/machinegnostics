# Fj — Gnostic Infidelity / Complement Activation

Gnostic complement to Fi measuring deviation / infidelity relative to learned concept center `z0` with scale `S`. Value range: `[1, ∞)`. 

---

## Overview

Evaluates `fj = cosh(2θ)` form for infidelity / residual magnitude measurement.

The centered coordinate is `θ = (x - z0) / S`.

---

## Key Features

- Learnable `z0` and `S`
- Complement to `Fi`
- Gnostic characteristic via custom autograd

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
- `last_output`: Most recent infidelity response

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
from machinegnostics.magnet import Fj

activation = Fj()
result = activation(np.array([[0.0]]))
print(result.data.tolist())  # [[1.0]]
```

---

**Author:** Nirmal Parmar
