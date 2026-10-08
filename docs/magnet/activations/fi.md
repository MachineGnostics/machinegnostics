# Fi — Gnostic Fidelity Activation

The `Fi` activation measures concept fidelity using the gnostic characteristic `fi = sech(2θ)` with learnable center `z0` and scale `S`. Value range: `(0, 1]`. 

---

## Overview

`Fi` maps input into a centered coordinate `θ = (x - z0) / S` and evaluates a bell-shaped fidelity response that peaks at `1` when the input matches the learned concept center and decays toward `0` as deviation grows. The learnable parameters allow the layer to align with data geometry during training.

---

## Key Features

- Learnable concept center (`z0`) and scale (`S`)
- Bell-shaped fidelity response in `(0, 1]`
- Custom autograd kernel for efficient gradient computation
- Data-driven `z0` initialization from first forward pass if not provided

---

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `learnable_S` | `bool` | `True` | Optimize scale through gated raw parameter `S = 2σ(S_raw)` |
| `learnable_z0` | `bool` | `True` | Optimize concept center `z0` |
| `initial_S` | `float` | `1.0` | Initial positive scale |
| `initial_z0` | `float` \| `None` | `None` | Initial concept center; if `None`, initialized from data median |
| `name` | `str` \| `None` | `None` | Optional display name |
| `verbose` | `bool` | `False` | Verbose logging |

---

## Attributes

- `S`: Stored raw scale parameter tensor
- `z0`: Learnable concept center tensor
- `theta`: Most recent normalized deviation (`θ`)
- `last_output`: Most recent fidelity response

---

## Methods

### `forward(x, training=True)`

Transform inputs into estimating fidelity values `fi = sech(2θ)` with `θ = (x - z0) / S`.

- **x**: `Tensor` or array-like — Input features
- **training**: `bool` — Compatibility flag (same computation in training and inference)
- **Returns**: `Tensor` — Fidelity response in `(0, 1]`

---

## Example Usage

```python
import numpy as np
from machinegnostics.magnet import Fi

activation = Fi(learnable_S=True, learnable_z0=True, initial_S=1.0)
output = activation(np.array([[0.0, 0.5]]))
print(output.data.shape)  # (1, 2)
```

---

**Author:** Nirmal Parmar
