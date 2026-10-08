# Ei — Gnostic Entropy Activation

Entropy-like activation linking fidelity and residual uncertainty. Case `i` gives `1 - fi`; case `j` gives `fj - 1`. Value range: case `'i'` → `[0, 1)`; case `'j'` → `[0, ∞)`. 

---

## Overview

Exposes MAGNET's entropy-style family with learnable `z0` and `S`. Case `'i'` estimates; `'j'` quantifies.

The centered coordinate is `θ = (x - z0) / S`.

---

## Key Features

- Two modes (`i` / `j`)
- Learnable `z0`, `S`
- Links fidelity and entropy

---

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
`learnable_S` | `bool` | `True` | Optimize scale
`learnable_z0` | `bool` | `True` | Optimize center
`initial_S` | `float` | `1.0` | Initial scale
`initial_z0` | `float \| None` | `None` | Initial center
`case` | `str` | `'i'` | Mode (`i` / `j`)
`name` | `str \| None` | `None` | Name
`verbose` | `bool` | `False` | Logging

---

## Attributes

- `S`: Stored raw scale parameter tensor
- `z0`: Learnable center tensor
- `theta`: Most recent centered deviation (`θ`)
- `case`: Active entropy formulation (`i` / `j`)
- `last_output`: Most recent entropy response

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
from machinegnostics.magnet import Ei

activation = Ei()
result = activation(np.array([[0.0, 1.0]]))
print(result.data.tolist())  # [[1.0, 0.0]]
```

---

**Author:** Nirmal Parmar
