# make_longley_check_data: Longley Economic Dataset

The `make_longley_check_data` function generates a Longley-like economic dataset. Ideally suited for testing numerical accuracy and stability in regression models, this dataset is famous for its high multicollinearity among predictors.

---

## Overview

This utility creates a synthetic version of the Longley economic dataset:

-   **Structure**: 16 observations with 7 economic variables.
-   **Characteristics**: High collinearity between predictors like `GNP`, `Population`, and `Year`.
-   **Purpose**: Validating the numerical stability of regression algorithms (e.g., least squares) under conditions of ill-conditioning.
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
| `data`         | numpy.ndarray | Array of shape `(16, 7)` containing economic indicators.                    |
| `column_names` | list[str]     | List of column names in order: `['GNP.deflator', 'GNP', 'Unemployed', 'Armed.Forces', 'Population', 'Year', 'Employed']`. |

---

## Example Usage

```python
from machinegnostics.data import make_longley_check_data

# Generate Longley data
data, cols = make_longley_check_data()

print(f"Shape: {data.shape}")
# Output: (16, 7)

print(f"Columns: {cols}")
# Output: ['GNP.deflator', 'GNP', 'Unemployed', 'Armed.Forces', 'Population', 'Year', 'Employed']
```
