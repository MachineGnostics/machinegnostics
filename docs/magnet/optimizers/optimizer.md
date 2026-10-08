# Optimizer — Abstract Base Optimizer Class

The foundational abstract base class for all MAGNET optimization algorithms.

---

## Overview

`Optimizer` is an abstract base class (ABC) that defines the interface for all optimization strategies in MAGNET. It provides common utilities for parameter updates, state management, and learning rate control, with specialized handling for gnostic parameters (`s`, `z0`, `*_s`, `*_z0`).

---

## Key Features

- **Abstract Interface**: Enforces consistent `step(param, grad)` signature across all optimizers
- **State Management**: `state_dict()` and `load_state_dict()` for serialization
- **Learning Rate Control**: `get_lr()` and `set_lr()` for dynamic adjustment
- **Gnostic Parameter Support**: Specialized scaling for `s`, `z0`, `*_s`, `*_z0` parameters

---

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `lr` | `float` | `0.001` | Base learning rate for parameter updates |

---

## Methods

### `step(param, grad, **kwargs)` *(abstract)*

Update parameter `param` using gradient `grad`. Must be implemented by all subclasses. For MAGNET parameters (`s`, `z0`, `*_s`, `*_z0`), applies specialized scaling.

### `state_dict()`

Return optimizer state (momentum vectors, step counts, etc.) for serialization.

### `load_state_dict(state)`

Restore optimizer state from a saved dictionary.

### `zero_grad()`

Zero out gradients for all parameters being optimized.

### `get_lr()`

Return current learning rate(s) for the optimizer.

### `set_lr(lr)`

Update the learning rate(s) for the optimizer.

---

## Example Usage

```python
from machinegnostics.magnet import Optimizer

class MyOptimizer(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__(lr=lr)

    def step(self, param, grad, **kwargs):
        # Simple gradient descent
        param -= self.lr * grad
        return param

opt = MyOptimizer(lr=0.01)
```

---

**Author:** Nirmal Parmar
