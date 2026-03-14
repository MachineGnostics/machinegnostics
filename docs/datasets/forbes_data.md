# make_forbes_check_data: Forbes Dataset (1857)

The `make_forbes_check_data` function retrieves Forbes' historic dataset on boiling points measured in the Alps. This dataset is a standard benchmark in robust statistics, often used to demonstrate the impact of outliers on regression models.

---

## Overview

Collected by James D. Forbes in the 19th century, this dataset contains 17 observations of the boiling point of water and barometric pressure at various altitudes in the Alps.

-   **Features ($X$)**: Boiling Point in degrees Fahrenheit ($^\circ$F).
-   **Target ($y$)**: Barometric Pressure in inches of mercury (inHg).
-   **Significance**: It is classically used to demonstrate robust regression because **Observation 12** is widely considered an outlier (likely due to a transcription or measurement error) that skews standard least-squares regression.
-   **Typical Usage**: Modeling the relationship: $100 \times \log_{10}(\text{Pressure}) \sim \text{BoilingPoint}$.

---

## Returns

| Return | Type          | Description                                    |
| :----- | :------------ | :--------------------------------------------- |
| `X`    | numpy.ndarray | Boiling Point ($^\circ$F). Shape `(17, 1)`.    |
| `y`    | numpy.ndarray | Pressure (inHg). Shape `(17,)`.                |

---

## Example Usage

```python
from machinegnostics.data import make_forbes_check_data
import numpy as np

# Load the dataset
X, y = make_forbes_check_data()

print(f"Dataset shape: X={X.shape}, y={y.shape}")
# Output: Dataset shape: X=(17, 1), y=(17,)

# Inspect the known outlier (Index 11, Observation 12)
print(f"Observation 12: Boiling Point={X[11,0]}, Pressure={y[11]}")
```

---

**Source:** Forbes, J. D. (1857). "Further experiments on the boiling point of water". Transactions of the Royal Society of Edinburgh. 21 (2): 235–243.

**Author:** Nirmal Parmar
