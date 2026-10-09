# Callback — Base Training Callback

Base interface for observing and influencing MAGNET training loops.

---

## Overview

`Callback` provides lifecycle hooks (`on_train_begin/end`, `on_epoch_begin/end`) that receive a mutable `logs` dictionary. Subclasses can inspect model state, trigger early stopping, or persist weights. `EarlyStopping` extends it with a concrete implementation.

---

## Key Features

- `set_model()` attaches the model before training
- `on_train_begin/end()` hooks for full-training lifecycle
- `on_epoch_begin/end()` hooks for per-epoch events
- `EarlyStopping` subclass for automatic training termination

---

## Methods

### `set_model(model)`

Attach the current training model.

### `on_train_begin(logs=None)`

Called once before training starts.

### `on_train_end(logs=None)`

Called once after training finishes.

### `on_epoch_begin(epoch, logs=None)`

Called at the beginning of each epoch.

### `on_epoch_end(epoch, logs=None)`

Called at the end of each epoch; most common hook.

---

## Example Usage

```python
from machinegnostics.magnet import Callback

class PrintCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch}: {logs}")
```

---

**Author:** Nirmal Parmar
