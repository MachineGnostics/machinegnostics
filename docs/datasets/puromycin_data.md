# make_puromycin_check_data: Puromycin Reaction Rate Dataset

The `make_puromycin_check_data` function generates a synthetic version of the Puromycin biochemical reaction dataset. This dataset is a classic benchmark for non-linear regression modeling, specifically for fitting Michaelis–Menten kinetics to reaction rates in treated versus untreated cells.

---

## Overview

This utility provides a reproducible, Puromycin-like dataset:

-   **Structure**: 23 observations with 3 variables.
-   **Groups**: Data distinguishes between treated (`state=1`) and untreated (`state=0`) samples.
-   **Relationships**: Follows Michaelis–Menten kinetics ($Velocity = \frac{V_{max} \cdot Concentration}{K_m + Concentration}$) with distinct parameters for each group.
-   **Purpose**: Ideal for testing non-linear regression models and analyzing group differences in curve fitting.
-   **Reproducibility**: Uses a fixed seed (default 42).

---

## Parameters

| Parameter | Type | Description | Default |
| :--- | :--- | :-------------------------------------------------------------------------- | :--- |
| `seed`    | int  | Random seed for reproducibility.                                            | `42` |

---

## Returns

| Return         | Type          | Description                                                                 |
| :------------- | :------------ | :-------------------------------------------------------------------------- |
| `data`         | numpy.ndarray | Array of shape `(23, 3)`.                                                   |
| `column_names` | list[str]     | List of column names: `['conc', 'rate', 'state']`.                          |

---

## Columns Description

| Column | Description |
| :--- | :--- |
| `conc` | Substrate concentration (ppm) |
| `rate` | Initial reaction velocity (counts/min/min) |
| `state` | Indicator of treatment: `0` = Untreated, `1` = Treated with Puromycin |

---

## Example Usage

```python
from machinegnostics.data import make_puromycin_check_data

# Generate Puromycin data
data, cols = make_puromycin_check_data()

# Separate groups
untreated = data[data[:, 2] == 0]
treated = data[data[:, 2] == 1]

print(f"Untreated samples: {len(untreated)}")
# Output: Untreated samples: 12 (approx)

print(f"Treated samples: {len(treated)}")
# Output: Treated samples: 11 (approx)
```
