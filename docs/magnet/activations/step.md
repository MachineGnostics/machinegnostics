# Step — Hard Threshold Activation

Maps values above threshold (default 0.0) to 1, others to 0. Mainly useful for explicit thresholding or post-processing.

---

## Overview

Not smooth; provides limited gradient information. Useful when a hard decision boundary is required rather than a differentiable transition.

---

## Key Features

- Hard binary output (0 / 1)
- Configurable `threshold`
- Limited gradient for training

---

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
`threshold` | `float` | `0.0` | Decision boundary
`name` | `str \| None` | `None` | Layer name
`verbose` | `bool` | `False` | Debug logging

---

## Methods

### `forward(x, training=True)`

Return the activation applied to input.

- **x**: `Tensor` or array-like — Input values of any shape
- **Returns**: `Tensor` with same shape as `x`

---

## Example Usage

```python
import numpy as np
from machinegnostics.magnet import Step

activation = Step()
result = activation(np.array([[0.25, 0.75]]))
print(result.data.tolist())  # [[0.0, 1.0]]
```

---

**Author:** Nirmal Parmar
