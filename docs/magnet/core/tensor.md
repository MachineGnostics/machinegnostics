# Tensor — Torch-Backed Tensor Facade

The `Tensor` class hides PyTorch autograd behind a MAGNET-friendly interface with NumPy-style inspection (`data`, `grad`, `shape`).

---

## Overview

`Tensor` is the only array object users interact with in MAGNET. It wraps `torch.Tensor`, exposes `.data` and `.grad` as NumPy arrays, and supports arithmetic, matrix operations, `backward()`, and device/dtype selection.

---

## Key Features

- `data` / `grad` as NumPy arrays
- `shape`, `ndim`, `size`, `device`, `dtype` properties
- Arithmetic operators (`+`, `-`, `*`, `/`, `**`, `@`)
- `backward()`, `zero_grad()`, `clone()`, `detach()`
- `reshape()`, `transpose()`, `sum()`, `mean()`, `exp()`, `log()`, `sigmoid()`, `relu()`, `tanh()`
- `to()` for device/dtype conversion
- `__array__` interface for NumPy interoperability

---

## Example Usage

```python
import numpy as np
from machinegnostics.magnet import Tensor

t = Tensor(np.array([[1.0, 2.0]]), requires_grad=True)
print(t.shape)       # (1, 2)
print(t.data)        # [[1. 2.]]
```

---

**Author:** Nirmal Parmar  
