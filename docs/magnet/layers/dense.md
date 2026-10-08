# Dense — Fully Connected Layer

The `Dense` layer applies a trainable affine transformation `y = x @ W + b` to the input.

---

## Overview

`Dense` is MAGNET's standard fully connected layer. It learns a weight matrix `W` and optional bias vector `b`, supporting both current and older compatibility initializer APIs.

---

## Key Features

- Standard affine transformation on last input axis
- Configurable `in_features` / `out_features`
- Optional bias (`use_bias=True`)
- Compatibility initializers (`kernel_initializer`, `bias_initializer`)
- Custom backward function support

---

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `in_features` | `int` | `None` | Input dimension (alias `n_in`) |
| `out_features` | `int` | `None` | Output dimension (alias `n_out`) |
| `weight_init` / `kernel_initializer` | callable | `XavierUniform` | Weight initializer |
| `bias_init` / `bias_initializer` | callable | `Zeros()` | Bias initializer |
| `use_bias` | `bool` | `True` | Include bias term |
| `name` | `str` \| `None` | `None` | Layer name |
| `verbose` | `bool` | `False` | Verbose logging |

---

## Example Usage

```python
import numpy as np
from machinegnostics.magnet import Dense

d = Dense(in_features=3, out_features=2)
print(d(np.array([[1.0, 2.0, 3.0]])).shape)  # (1, 2)
```

---

**Author:** Nirmal Parmar  
