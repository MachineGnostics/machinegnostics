# BatchNorm — Batch Normalization Layer

Standard batch normalization layer for feature vectors.

---

## Overview

`BatchNorm` normalizes each feature dimension using batch statistics during training (`mean`, `var`) and running statistics (`running_mean`, `running_var`) during inference. It learns `gamma` (scale) and `beta` (shift) parameters. It is the standard normalization layer for stable optimization.

---

## Key Features

- Batch statistics during training; running statistics during inference
- Trainable `gamma` and `beta` parameters
- Running mean/variance tracking with `momentum`
- Numerical stability via `eps`

---

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `num_features` | `int` | Required | Number of feature dimensions |
| `momentum` | `float` | `0.9` | Running statistics update rate |
| `eps` | `float` | `1e-5` | Numerical stability constant |

---

## Methods

### `forward(x, training=True)`

Normalize using batch stats (`training=True`) or running stats (`training=False`). Updates running stats when training.

---

## Example Usage

```python
from machinegnostics.magnet import BatchNorm
layer = BatchNorm(num_features=3)
print(layer(np.ones((2, 3))).shape)  # (2, 3)
```

---

**Author:** Nirmal Parmar
