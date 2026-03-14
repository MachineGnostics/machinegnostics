# make_stackloss_check_data: Stack Loss Dataset

The `make_stackloss_check_data` function retrieves the classic Stack Loss dataset (Brownlee, 1965). This dataset describes the operation of a plant for the oxidation of ammonia to nitric acid and is a standard benchmark for robust regression due to the presence of well-known outliers.

---

## Overview

The dataset consists of 21 operational days of a plant converting ammonia to nitric acid. The goal is typically to predict the `Stack Loss` (the amount of ammonia escaping unabsorbed) based on three operational variables.

-   **Significance**: Ideally suited for demonstrating robust regression methods because it contains several acknowledged outliers (observations 1, 2, 3, and 21) that can distort standard least-squares models.
-   **Size**: 21 samples, 4 variables.

---

## Data Dictionary

The dataset is returned as a single matrix with the following columns:

1.  **Air Flow** (Feature): Rate of operation of the plant.
2.  **Water Temp.** (Feature): Cooling water inlet temperature.
3.  **Acid Conc.** (Feature): Acid concentration (in per 1000 minus 500).
4.  **Stack.Loss** (Target): Amount of ammonia escaping the absorption column.

---

## Returns

| Return         | Type          | Description                                                    |
| :------------- | :------------ | :------------------------------------------------------------- |
| `data`         | numpy.ndarray | The complete data array of shape `(21, 4)`.                    |
| `column_names` | list of str   | The list of columns: `['Air Flow', 'Water Temp.', 'Acid Conc.', 'Stack.Loss']` |

---

## Example Usage

```python
from machinegnostics.data import make_stackloss_check_data
import numpy as np

# Load the dataset
data, names = make_stackloss_check_data()

print(f"Data shape: {data.shape}")
print(f"Columns: {names}")

# Separate Features (X) and Target (y)
X = data[:, :3]
y = data[:, 3]
```

---

**Source:** Brownlee, K. A. (1965). Statistical Theory and Methodology in Science and Engineering. New York: John Wiley & Sons.

**Author:** Nirmal Parmar
