# Initializer — Abstract Base Initializer Class

The foundational abstract base class for all MAGNET weight initializers.

---

## Overview

`Initializer` is an abstract base class (ABC) that defines the interface for all weight initialization strategies in MAGNET. It provides common utilities for shape validation and fan-in/fan-out computation, and requires subclasses to implement the `__call__` method for generating weight arrays.

---

## Key Features

- **Abstract Interface**: Enforces consistent `__call__(shape)` signature across all initializers
- **Shape Normalization**: Helper method `normalize_shape()` for dimension validation
- **Fan Computation**: `compute_fans()` utility for calculating input/output dimensions
- **Dtype Control**: Configurable output data type (default `np.float64`)

---

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `dtype` | `np.dtype` | `np.float64` | Output data type for generated arrays |

---

## Methods

### `__call__(shape)` *(abstract)*

Generate and return a weight array of the given shape. Must be implemented by all subclasses.

### `normalize_shape(shape)`

Validate and normalize the shape tuple, raising errors for invalid inputs.

### `compute_fans(shape)`

Compute fan-in and fan-out dimensions from a shape tuple, useful for Glorot/He initializers.

---

## Example Usage

```python
from machinegnostics.magnet import Initializer

class MyInitializer(Initializer):
    def __call__(self, shape):
        shape = self.normalize_shape(shape)
        return np.ones(shape, dtype=self.dtype)

init = MyInitializer()
W = init((10, 20))
```

---

**Author:** Nirmal Parmar
