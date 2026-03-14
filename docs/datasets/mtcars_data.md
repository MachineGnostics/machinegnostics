# make_mtcars_check_data: MtCars Dataset

The `make_mtcars_check_data` function generates a synthetic version of the famous Motor Trend Car Road Tests (MtCars) dataset. This dataset is widely used for teaching and testing linear regression concepts, particularly for predicting fuel efficiency (`mpg`) based on vehicle characteristics.

---

## Overview

This utility provides a reproducible, MtCars-shaped dataset:

-   **Structure**: 32 observations (cars) with 11 variables.
-   **Relationships**: Realistic correlations, such as `mpg` decreasing as `hp` (horsepower) and `wt` (weight) increase.
-   **Purpose**: Excellent for testing regression models, feature selection, and data visualization techniques.
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
| `data`         | numpy.ndarray | Array of shape `(32, 11)` containing vehicle specifications.                |
| `column_names` | list[str]     | List of column names: `['mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']`. |
| `car_names`    | list[str]     | List of placeholder car names (e.g., `'Car 1'`, `'Car 2'`) matching the rows. |

---

## Columns Description

| Column | Description |
| :--- | :--- |
| `mpg` | Miles/(US) gallon |
| `cyl` | Number of cylinders |
| `disp` | Displacement (cu.in.) |
| `hp` | Gross horsepower |
| `drat` | Rear axle ratio |
| `wt` | Weight (1000 lbs) |
| `qsec` | 1/4 mile time |
| `vs` | Engine (0 = V-shaped, 1 = straight) |
| `am` | Transmission (0 = automatic, 1 = manual) |
| `gear` | Number of forward gears |
| `carb` | Number of carburetors |

---

## Example Usage

```python
from machinegnostics.data import make_mtcars_check_data

# Generate MtCars data
data, cols, names = make_mtcars_check_data()

print(f"Shape: {data.shape}")
# Output: (32, 11)

print(f"Key Features: {cols[3]} (HP), {cols[5]} (Weight)")
# Output: Key Features: hp (HP), wt (Weight)

# Predict MPG (column 0)
X = data[:, [3, 5]] # hp, wt
y = data[:, 0]      # mpg
```
