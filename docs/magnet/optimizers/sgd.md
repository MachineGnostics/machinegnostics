# SGD — Stochastic Gradient Descent Optimizer

The classic gradient descent optimizer with optional momentum for accelerated convergence.

---

## Overview

`SGD` (Stochastic Gradient Descent) is the foundational optimization algorithm that updates parameters by moving in the direction of the negative gradient. For MAGNET models, SGD provides specialized parameter scaling for `s`, `z0`, `*_s`, and `*_z0` parameters, ensuring proper gradient-based optimization of gnostic transformations.

---

## Key Features

- **Classic Gradient Descent**: Direct parameter updates based on gradient direction
- **Momentum Support**: Optional momentum term for accelerated convergence and reduced oscillation
- **Magnet Integration**: Specialized scaling for MAGNET parameters (`s`, `z0`, `*_s`, `*_z0`)
- **Learning Rate Control**: Configurable base learning rate for parameter updates
- **Nesterov Momentum**: Optional Nesterov-accelerated momentum for improved convergence

---

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `lr` | `float` | `0.01` | Base learning rate for parameter updates |
| `momentum` | `float` | `0.0` | Momentum factor for accelerated updates |
| `dampening` | `float` | `0.0` | Dampening factor for momentum updates |
| `nesterov` | `bool` | `False` | Enable Nesterov-accelerated momentum |
| `weight_decay` | `float` | `0.0` | L2 regularization coefficient (weight decay) |

---

## Methods

### `__init__(lr=0.01, momentum=0.0, dampening=0.0, nesterov=False, weight_decay=0.0)`

Initialize the SGD optimizer with specified hyperparameters.

### `step(param, grad, **kwargs)`

Update parameter `param` using gradient `grad`. For MAGNET parameters (`s`, `z0`, `*_s`, `*_z0`), applies specialized scaling based on their parameter type. Includes momentum updates if configured.

### `state_dict()`

Return optimizer state (velocity vector, step count) for serialization.

### `load_state_dict(state)`

Restore optimizer state from a saved dictionary.

### `zero_grad()`

Zero out gradients for all parameters being optimized.

### `get_lr()`

Return current learning rate for the optimizer.

### `set_lr(lr)`

Update the learning rate for the optimizer.

---

## Example Usage

```python
from machinegnostics.magnet import Sequential, Dense, ReLU, MSE, SGD
import numpy as np

model = Sequential(layers=[
    Dense(in_features=4, out_features=8), ReLU(),
    Dense(in_features=8, out_features=1)
])

model.compile(loss=MSE(), optimizer=SGD(lr=0.001))
model.fit(X_train, y_train, epochs=10, batch_size=32)
loss, metric = model.evaluate(X_val, y_val)
print(f"Validation Loss: {loss:.4f}")
```


**Author:** Nirmal Parmar
