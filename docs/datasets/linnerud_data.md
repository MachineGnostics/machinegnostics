# make_linnerud_check_data: Linnerud Multi-output Regression Dataset

The `make_linnerud_check_data` function loads or generates a Linnerud-like multi-output regression dataset. This dataset consists of 3 exercise data variables and 3 physiological variables collected from 20 middle-aged men in a fitness club. It is a classic dataset for multi-output regression tasks.

---

## Overview

This utility provides access to the Linnerud dataset:

-   **Features ($X$)**: Three exercise variables: `Chins`, `Situps`, and `Jumps`.
-   **Targets ($Y$)**: Three physiological variables: `Weight`, `Waist`, and `Pulse`.
-   **Purpose**: Ideal for testing models that predict multiple targets simultaneously (multi-output regression).
-   **Source**: Wraps Scikit-learn's `load_linnerud` if available; otherwise falls back to a similarly shaped synthetic generator.

---

## Parameters

| Parameter      | Type | Description                                      | Default |
| :------------- | :--- | :----------------------------------------------- | :------ |
| `return_names` | bool | If `True`, returns the list of column names for $X$ and $Y$. | `True`  |

---

## Returns

| Return    | Type          | Description                                                                 |
| :-------- | :------------ | :-------------------------------------------------------------------------- |
| `X`       | numpy.ndarray | Exercise features of shape `(20, 3)`. Columns correspond to `Chins`, `Situps`, `Jumps`. |
| `Y`       | numpy.ndarray | Physiological targets of shape `(20, 3)`. Columns correspond to `Weight`, `Waist`, `Pulse`. |
| `X_names` | list[str]     | Names of the feature columns `['Chins', 'Situps', 'Jumps']`. Returned only if `return_names=True`. |
| `Y_names` | list[str]     | Names of the target columns `['Weight', 'Waist', 'Pulse']`. Returned only if `return_names=True`. |

---

## Example Usage

```python
from machinegnostics.data import make_linnerud_check_data

# Load data with column names
X, Y, X_names, Y_names = make_linnerud_check_data(return_names=True)

print(f"Features: {X_names}")
# Output: ['Chins', 'Situps', 'Jumps']

print(f"Targets: {Y_names}")
# Output: ['Weight', 'Waist', 'Pulse']

print(f"X Shape: {X.shape}, Y Shape: {Y.shape}")
# Output: X Shape: (20, 3), Y Shape: (20, 3)
```
