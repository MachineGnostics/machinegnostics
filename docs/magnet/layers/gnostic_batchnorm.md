# GnosticBatchNorm — Gnostic Batch Normalization Layer

Batch normalization with an additional gnostic weighting term applied to centered data.

---

## Overview

`GnosticBatchNorm` extends `BatchNorm` by computing a gnostic weighting tensor from the centered batch data. It selects between estimating (`"i"`) and quantifying (`"j"`) weights via the `kind` parameter, then multiplies that weight into the normalized output. The scale parameter `S` and `lel` are passed through to the gnostic weight computation.

---

## Key Features

- Standard batch normalization with gnostic weighting
- `kind="i"` → estimating weights; `kind="j"` → quantifying weights
- Configurable scale `S` and `lel`
- Training/inference mode handling

---

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `num_features` | `int` | Required | Feature count |
| `S` | `float` \| `str` | `2.0` | Scale for gnostic weights |
| `kind` | `str` | `"i"` | `"i"` (estimating) or `"j"` (quantifying) |
| `momentum` | `float` | `0.9` | Running stats momentum |
| `eps` | `float` | `1e-5` | Numerical stability |

---

## Methods

### `forward(x, training=True)`

Normalize batch, compute gnostic weights from centered data, apply weighting to normalized output, update running stats.

---

## Example Usage

```python
from machinegnostics.magnet import GnosticBatchNorm
layer = GnosticBatchNorm(num_features=3, kind="i", S=2.0)
```

---

**Author:** Nirmal Parmar
