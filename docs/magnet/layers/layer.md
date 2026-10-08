# Layer — Abstract Base Layer Class

The foundational contract for all MAGNET layer components.

---

## Overview

`Layer` defines the shared interface for every MAGNET building block: trainable parameters in `params`, gradient cache in `grads`, a forward pass method, an optional backward hook, and callable function behavior (`layer(x)`). It does not perform any computation on its own; subclasses implement `forward()` and optionally `backward()`.

---

## Key Features

- **Parameter Storage**: `params` dictionary holds trainable `Tensor` objects
- **Gradient Cache**: `grads` dictionary for manual or autograd-based gradients
- **Callable Interface**: Call `layer(x, training=True)` directly
- **Mode Tracking**: `set_mode()` records training/inference state
- **Parameter Iterator**: `parameters()` yields trainable tensors to optimizers
- **Gradient Sync**: `sync_grads()` copies tensor gradients for inspection

---

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `name` | `str` \| `None` | `None` | Display name; defaults to class name |
| `verbose` | `bool` | `False` | Enable debug-level logging |

---

## Attributes

| Attribute | Type | Description |
| :--- | :--- | :--- | :--- |
| `params` | `dict` | Trainable `Tensor` objects |
| `grads` | `dict` | Gradient arrays |
| `trainable` | `bool` | Whether parameters are trainable |
| `_training` | `bool` | Current mode (`True` = training) |

---

## Methods

### `forward(x, training=True)` *(abstract)*

Transform input in the forward pass. Subclasses must override.

### `backward(grad_output)` *(optional)*

Backward pass hook for manual gradient computation.

### `__call__(x, training=True)`

Alias for `forward()` so layers can be called like functions.

### `parameters()`

Yield all trainable tensors owned by the layer.

### `sync_grads()`

Copy tensor gradients into `grads` for inspection.

### `get_params_and_grads()`

Yield `(parameter, gradient)` pairs for optimizers.

### `set_mode(training)`

Record whether the layer is in training or inference mode.

---

## Example Usage

```python
from machinegnostics.magnet import Layer, Tensor
import numpy as np

class Scale(Layer):
    def __init__(self):
        super().__init__()
        self.params["w"] = Tensor(np.array([2.0]), requires_grad=True)
    def forward(self, x, training=True):
        x = x if isinstance(x, Tensor) else Tensor(x)
        return x * self.params["w"]
    def backward(self, grad_output):
        raise NotImplementedError

layer = Scale()
print(layer(np.array([3.0])).data)  # [6.0]
```

---

**Author:** Nirmal Parmar
