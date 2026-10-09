# Sequential — Linear Stack of Layers

Creates a linear pipeline of layers where output from each feeds directly into the next.

---

## Overview

`Sequential` is a `Model` subclass that represents a straight‑line neural network composed of stacked `Layer` objects. It provides a convenient constructor for specifying layer sequences and automatically handles weight management, forward propagation, and the training/evaluation lifecycle via its inherited `Model` interface.

---

## Key Features

- **Linear Architecture**: Layer stack where each output directly feeds the next
- **Flexible Construction**: Build networks with lists, tuples, or keyword‑style arguments
- **Model Integration**: Full support for `compile()`, `fit()`, `predict()`, `evaluate()`, `summary()`
- **Gnostic Support**: Seamless handling of MAGNET‑specific parameters (`s`, `z0`, `*_s`, `*_z0`) across layers
- **Layer Listeners**: Optional hooks for each layer in the stack for monitoring or custom behavior

---

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `layers` | `list[Layer]` | Required | Ordered list of child `Layer` objects forming the linear stack |

---

## Methods

### `compile(loss=None, optimizer=None)`

Configure the sequential model with a loss function and optimizer for training.

### `fit(X, y, epochs=1, batch_size=32, verbose=True, callbacks=None)`

Train the sequential stack on `(X, y)` data using the configured loss and optimizer, supporting mini‑batch SGD with callback hooks.

### `predict(X)`

Generate forward predictions for input `X` by propagating through the entire layer sequence.

### `evaluate(X, y)`

Compute loss and metric values on `(X, y)` data for validation.

### `summary()`

Print a formatted architecture summary showing layer types, output shapes, and parameter counts.

### `add(layer)`

Append a new `Layer` to the sequential stack, dynamically extending the model.

### `pop()`

Remove the last layer from the sequential stack and return it.

---

## Example Usage

```python
from machinegnostics.magnet import Sequential, Dense, ReLU, MSE

# Build a sequential model with MAGNET layers
model = Sequential(layers=[
    Dense(in_features=128, out_features=64), ReLU(),
    Dense(in_features=64, out_features=1)
])

# Compile model
model.compile(
    loss=MSE(),
    optimizer=Adam()
)

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=True)

# Evaluate performance
val_loss, val_metric = model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss:.4f}")

# Generate predictions
predictions = model.predict(X_test)
```

---

**Author:** Nirmal Parmar
