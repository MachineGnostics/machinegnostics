# RMSprop — Root Mean Square Propagation Optimizer

An adaptive learning rate method that divides the learning rate by a running average of the squared gradients.

---

## Overview

`RMSprop` (Root Mean Square Propagation) is an adaptive optimization algorithm that divides the learning rate by a running average of the squared gradients for each parameter. This helps with non‑convex optimization problems and provides better convergence for models with varying curvature. For MAGNET models, RMSprop provides specialized parameter scaling for `s`, `z0`, `*_s`, and `*_z0` parameters.

---

## Key Features

- **Adaptive Learning Rates**: Per‑parameter learning rates based on squared gradient history
- **Moving Average**: Exponential moving average of squared gradients for curvature adaptation
- **Magnet Integration**: Specialized scaling for MAGNET parameters (`s`, `z0`, `*_s`, `*_z0`)
- **Alpha Control**: Decay rate for the moving average of squared gradients
- **Eps Control**: Numerical stability parameter for denominator calculations

---

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `lr` | `float` | `0.001` | Base learning rate for parameter updates |
| `alpha` | `float` | `0.99` | Decay rate for the moving average of squared gradients |
| `eps` | `float` | `1e-8` | Small constant for numerical stability |
| `weight_decay` | `float` | `0.0` | L2 regularization coefficient (weight decay) |

---

## Methods

### `__init__(lr=0.001, alpha=0.99, eps=1e-8, weight_decay=0.0)`

Initialize the RMSprop optimizer with specified hyperparameters.

### `step(param, grad, **kwargs)`

Update parameter `param` using gradient `grad`. For MAGNET parameters (`s`, `z0`, `*_s`, `*_z0`), applies specialized scaling based on their parameter type.

### `state_dict()`

Return optimizer state (squared gradient moving average, step count) for serialization.

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
from machinegnostics.magnet import Sequential, Dense, ReLU, MSE, RMSprop
import numpy as np

model = Sequential(layers=[
    Dense(in_features=4, out_features=8), ReLU(),
    Dense(in_features=8, out_features=1)
])

model.compile(loss=MSE(), optimizer=RMSprop(lr=0.001))
model.fit(X_train, y_train, epochs=10, batch_size=32)
loss, metric = model.evaluate(X_val, y_val)
print(f"Validation Loss: {loss:.4f}")
```


**Author:** Nirmal Parmar
