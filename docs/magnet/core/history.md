# History — Training History Container

A dictionary-like container that tracks epoch-level metrics during MAGNET training.

---

## Overview

`History` extends Python's `dict` to store lists of metric values recorded at each training step or epoch. It provides `record()` to append new log entries and `last()` to retrieve the most recent value for a given metric key. It is automatically integrated with `Model.fit()` and used by the built-in callback system.

---

## Key Features

- Dictionary-like storage of metric lists
- `record()` for appending metric dictionaries
- `last()` for retrieving the latest value of a key
- Optional `verbose` logging

---

## Methods

### `record(logs)`

Append a dictionary of metric values. Each key gets a new entry in its list.

### `last(key, default=None)`

Return the last recorded value for the metric key, or `default` if empty.

---

## Example Usage

```python
from machinegnostics.magnet import History
history = History()
history.record({"loss": 0.5, "val_loss": 0.45})
print(history.last("loss"))  # 0.5
```

---

**Author:** Nirmal Parmar
