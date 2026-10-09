# Adam — Adaptive Moment Estimation Optimizer

An adaptive optimizer that maintains per‑parameter learning rates based on estimates of first and second moments of gradients.

---

## Overview

`Adam` (Adaptive Moment Estimation) is an adaptive optimization algorithm that computes adaptive learning rates for each parameter by estimating first and second moments of the gradients. It combines the advantages of AdaGrad (handles sparse gradients well) and RMSProp (works well in non‑convex optimization). For MAGNET models, Adam provides specialized parameter scaling for `s`, `z0`, `*_s`, and `*_z0` parameters.

---

## Key Features

- **Per‑Parameter Learning Rates**: Adaptive rates for each parameter based on gradient moments
- **Bias Correction**: Corrects bias in moment estimates during initial training steps
- **Magnet Integration**: Specialized scaling for MAGNET parameters (`s`, `z0`, `*_s`, `*_z0`)
- **AMSGrad**: Optional variant that maintains maximum of past squared gradients for convergence guarantees
- **Epsilon Control**: Numerical stability parameter for denominator calculations

---

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `lr` | `float` | `0.001` | Base learning rate for parameter updates |
| `beta1` | `float` | `0.9` | Exponential decay rate for first moment estimates |
| `beta2` | `float` | `0.999` | Exponential decay rate for second moment estimates |
| `eps` | `float` | `1e-8` | Small constant for numerical stability |
| `amsgrad` | `bool` | `False` | Enable AMSGrad variant for improved convergence |
| `weight_decay` | `float` | `0.0` | L2 regularization coefficient (weight decay) |

---

## Methods

### `__init__(lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, amsgrad=False, weight_decay=0.0)`

Initialize the Adam optimizer with specified hyperparameters.

### `step(param, grad, **kwargs)`

Update parameter `param` using gradient `grad`. For MAGNET parameters (`s`, `z0`, `*_s`, `*_z0`), applies specialized scaling based on their parameter type.

### `state_dict()`

Return optimizer state (first/second moment vectors, step count) for serialization.

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
from machinegnostics.magnet import Sequential, Dense, ReLU, MSE, Adam
import numpy as np

model = Sequential(layers=[
    Dense(in_features=4, out_features=8), ReLU(),
    Dense(in_features=8, out_features=1)
])

model.compile(loss=MSE(), optimizer=Adam(lr=0.001))
model.fit(X_train, y_train, epochs=10, batch_size=32)
loss, metric = model.evaluate(X_val, y_val)
print(f"Validation Loss: {loss:.4f}")
```


**Author:** Nirmal Parmar
