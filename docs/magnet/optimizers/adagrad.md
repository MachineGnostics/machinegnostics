# Adagrad — Adaptive Gradient Descent Optimizer

An adaptive optimizer that accumulates squared gradients for each parameter, scaling the learning rate inversely with the magnitude of past gradients.

---

## Overview

`Adagrad` (Adaptive Gradient Descent) is an adaptive optimization algorithm that maintains a running sum of squared gradients for each parameter, automatically scaling the learning rate inversely with the magnitude of past gradients. This helps with sparse data and handles parameters that receive infrequent updates. For MAGNET models, Adagrad provides specialized parameter scaling for `s`, `z0`, `*_s`, and `*_z0` parameters.

---

## Key Features

- **Adaptive Learning Rates**: Per‑parameter learning rates based on accumulated squared gradients
- **Sqrt Scaling**: Learning rate scaling by the square root of accumulated gradients
- **Magnet Integration**: Specialized scaling for MAGNET parameters (`s`, `z0`, `*_s`, `*_z0`)
- **Epsilon Control**: Numerical stability parameter for denominator calculations
- **Learning Rate Decay**: Optional learning rate decay over time

---

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `lr` | `float` | `0.01` | Base learning rate for parameter updates |
| `eps` | `float` | `1e-8` | Small constant for numerical stability |
| `weight_decay` | `float` | `0.0` | L2 regularization coefficient (weight decay) |
| `lr_decay` | `float` | `0.0` | Learning rate decay factor (per step) |

---

## Methods

### `__init__(lr=0.01, eps=1e-8, weight_decay=0.0, lr_decay=0.0)`

Initialize the Adagrad optimizer with specified hyperparameters.

### `step(param, grad, **kwargs)`

Update parameter `param` using gradient `grad`. For MAGNET parameters (`s`, `z0`, `*_s`, `*_z0`), applies specialized scaling based on their parameter type. The update uses the accumulated squared gradient history.

### `state_dict()`

Return optimizer state (accumulated squared gradients, step count) for serialization.

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
from machinegnostics.magnet import Sequential, Dense, ReLU, MSE, Adagrad
import numpy as np

model = Sequential(layers=[
    Dense(in_features=4, out_features=8), ReLU(),
    Dense(in_features=8, out_features=1)
])

model.compile(loss=MSE(), optimizer=Adagrad(lr=0.001))
model.fit(X_train, y_train, epochs=10, batch_size=32)
loss, metric = model.evaluate(X_val, y_val)
print(f"Validation Loss: {loss:.4f}")
```


**Author:** Nirmal Parmar
