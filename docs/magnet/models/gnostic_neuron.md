# GnosticNeuron — MAGNET‑Specific Gnostic Algebra Neuron

Implements a deep‑learning neuron grounded in gnostic algebra with characteristic activation transformations and parameter adaptations.

---

## Overview

`GnosticNeuron` is a MAGNET‑specific neural network component that operates on the gnostic algebra framework. It introduces a centered coordinate system with parameters `s` (scale) and `z0` (center), and supports both forward and backward gnostic transformations through activation functions `fi`, `fj`, `hi`, `hj`, and `ei`. The neuron maintains the gnostic characteristics:

- `fi = sech(2θ)`  (fidelity output)
- `fj = cosh(2θ)`  (infidelity entropy)
- `hi = tanh(2θ)`  (relevance output)
- `hj = sinh(2θ)`  (irrelevance output)
- `ei = 1 - fi`    (gnostic entropy) 

Centered coordinate: `θ = (x - z0) / S`, where `S` is the scale parameter and `z0` is the center.

This neuron integrates seamlessly into MAGNET models, providing the core gnostic transformations for advanced deep‑learning applications.

---

## Key Features

- **Gnostic Algebra Core**: Forward/backward passes using `fi`, `fj`, `hi`, `hj`, and `ei` transformations
- **Centered Coordinate System**: `θ = (x - z0) / S` with `s` (scale) and `z0` (center) parameters
- **Dual‑Entropy Types**: Configurable entropy output (`'i'` → `[0,1)`, `'j'` → `[0,∞)`) via `ei` parameter
- **Adaptive Optimizers**: Integrated handling of `s` and `z0` parameters by optimizers with specialized scaling
- **Flexible Activation Types**: Works with both standard activations and gnostic versions
- **Parameter Management**: Dedicated handling of `s`, `z0`, `*_s`, and `*_z0` parameters through MAGNET optimizers

---

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `activation` | `str` \| `callable` | `'relu'` | Activation function name or callable; supports MAGNET‑specific gnostic activations |
| `units` | `int` | `1` | Number of output units/neurons |
| `s` | `float` | `1.0` | Scale parameter controlling coordinate spread `S` (used in `θ = (x - z0) / S`) |
| `z0` | `float` | `0.0` | Center parameter defining the origin of the centered coordinate system |
| `use_gnostic` | `bool` | `True` | Enable MAGNET‑specific gnostic transformations if `True` |

---

## Methods

### `__init__(activation='relu', units=1, s=1.0, z0=0.0, use_gnostic=True)`

Initialize a gnostic neuron with the specified activation, units, and gnostic parameters.

### `__call__(X, training=True)`

Perform forward pass with gnostic transformations. When `use_gnostic` is `True`, applies the full gnostic algebra framework with `fi`, `fj`, `hi`, `hj`, and `ei` transformations using centered coordinates `θ = (x - z0) / s`. Otherwise, behaves like a standard neuron with the specified activation.

### `compile(loss=None, optimizer=None)`

Configure the neuron (or model containing the neuron) with a loss function and optimizer, ensuring proper handling of `s` and `z0` parameters.

### `fit(X, y, epochs=1, batch_size=32, verbose=True, callbacks=None)`

Train the neuron on `(X, y)` data using the configured loss and optimizer, with specialized parameter scaling for `s` and `z0` during updates.

### `predict(X)`

Generate forward predictions for input `X` without any training updates, applying gnostic transformations if enabled.

### `evaluate(X, y)`

Compute loss and metric values on `(X, y)` data, returning both for validation while preserving gnostic parameter adaptations.

### `get_weights()`

Collect all trainable parameters from the neuron, including `s`, `z0`, and activation weights.

### `set_weights(weights)`

Restore neuron weights from a flat list, ensuring proper ordering of `s`, `z0`, and activation parameters.

---

## Example Usage

```python
from machinegnostics.magnet import GnosticNeuron, MSE, SGD

# Create a gnostic neuron with MAGNET parameters
neuron = GnosticNeuron(
    activation='relu',
    units=1,
    s=1.5,
    z0=0.2,
    use_gnostic=True
)

# Compile neuron (part of a model)
neuron.compile(
    loss=MSE(),
    optimizer=SGD()
)

# Train the neuron
neuron.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate performance
loss, metric = neuron.evaluate(X_val, y_val)
print(f"Validation Loss: {loss:.4f}")

# Generate predictions
preds = neuron.predict(X_test)
```

---

**Author:** Nirmal Parmar
