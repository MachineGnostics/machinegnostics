# Model — Abstract Base Model Class

The foundational container for MAGNET models: manages weights, layers, and training state across forward/backward passes.

---

## Overview

`Model` serves as the abstract base class for all MAGNET neural network components. It implements the core `compile()`/`fit()`/`predict()`/`evaluate()`/`summary()` lifecycle, maintains a stack of `Layer` objects with parameter and buffer management, and provides utilities for weight serialization, training history tracking, and model checkpointing. All MAGNET models—from simple `Sequential` stacks to specialized `GnosticNeuron` units—inherit this common interface.

---

## Key Features

- **Model Lifecycle**: `compile()`, `fit()`, `predict()`, `evaluate()`, `summary()` orchestration
- **Layer Stack**: Ordered stack of child `Layer` objects with flat parameter (`params`) and buffer (`buffers`) collections
- **Weight Management**: `get_weights()`/`set_weights()`, `save_weights()`/`load_weights()` APIs for serialization
- **Training History**: Automatic tracking of epoch‑level metrics via the built‑in `History` callback
- **Callbacks**: Pluggable training control (`EarlyStopping`, custom `Callback` subclasses) for flexible training loops
- **Gnostic Integration**: Parameter specializations for `s`, `z0`, `*_s`, and `*_z0` with adaptive scaling during optimizer updates

---

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `layers` | `list[Layer]` | `[]` | Initial list of child `Layer` objects |
| `loss` | `callable` | `None` | Loss function for compiled models |
| `optimizer` | `Optimizer` | `None` | Optimizer instance for weight updates |

---

## Methods

### `compile(loss=None, optimizer=None)`

Configure the model with a loss function and optimizer for training.

### `fit(X, y, epochs=1, batch_size=32, verbose=True, callbacks=None)`

Train the model on `(X, y)` data using the configured loss and optimizer, supporting mini‑batch SGD with callback hooks.

### `predict(X)`

Generate forward predictions for input `X` without any training updates.

### `evaluate(X, y)`

Compute loss and metric values on `(X, y)` data, returning both for validation.

### `summary()`

Print a structured architecture summary showing layer stack, parameter counts, and trainable status.

### `get_weights()`

Collect all trainable parameters from child layers into a flat list of numpy arrays.

### `set_weights(weights)`

Restore model weights from a flat list of numpy arrays, ensuring shape compatibility.

### `save_weights(filepath)`

Persist model parameters to disk in a platform‑independent format.

### `load_weights(filepath)`

Restore model parameters from a saved file.

### `add_weight(shape, dtype=np.float64, name=None, initializer='glorot_uniform', trainable=True)`

Add a new trainable weight tensor to the model, with flexible initialization and Gnostic‑aware parameter naming.

### `add_buffer(shape, dtype=np.float64, name=None, initializer='zeros')`

Add a non‑trainable buffer tensor (such as running statistics), used for temporary storage during forward/backward passes.

---

## Example Usage

```python
from machinegnostics.magnet import Sequential, Dense, ReLU, SGD

# Sequential model with two hidden layers
model = Sequential(layers=[
    Dense(in_features=64, out_features=32), ReLU(),
    Dense(in_features=32, out_features=1)
])

# Compile model
model.compile(
    loss=MSE(),
    optimizer=SGD()
)

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate performance
loss, metric = model.evaluate(X_val, y_val)
print(f"Validation Loss: {loss:.4f}")

# Make predictions
preds = model.predict(X_test)
```

---

**Author:** Nirmal Parmar
