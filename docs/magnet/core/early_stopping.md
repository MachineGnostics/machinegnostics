# EarlyStopping — Training Stopping Callback

Stops training when a monitored metric stops improving.

---

## Overview

`EarlyStopping` monitors a metric (typically `val_loss`), tracks the best value seen so far, and stops training after `patience` epochs without improvement. It can optionally restore the best observed weights when stopping.

---

## Key Features

- Monitor any metric by name (`monitor`)
- Configurable patience (`patience`) and minimum delta (`min_delta`)
- Optional best weight restoration (`restore_best_weights`)
- Verbose logging support

---

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `monitor` | `str` | `"val_loss"` | Metric name to watch |
| `patience` | `int` | `5` | Epochs without improvement before stopping |
| `min_delta` | `float` | `0.0` | Minimum improvement required |
| `restore_best_weights` | `bool` | `True` | Restore best weights when stopping |

---

## Methods

### `on_epoch_end(epoch, logs=None)`

Inspect logs; stop if metric hasn't improved for `patience` epochs.

---

## Example Usage

```python
from machinegnostics.magnet import EarlyStopping
stopper = EarlyStopping(monitor="val_loss", patience=3)
```

---

**Author:** Nirmal Parmar
