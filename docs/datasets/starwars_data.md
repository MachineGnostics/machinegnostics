# make_starwars_check_data: Star Wars Characters Dataset

The `make_starwars_check_data` function generates a synthetic Star Wars-like dataset containing demographics for 87 characters. Inspired by the `dplyr` dataset in R, this utility is perfect for practicing categorical analysis, grouping operations, and basic data exploration.

---

## Overview

This utility creates a character dataset similar to the original R dataset:

-   **Structure**: 87 observations (characters).
-   **Variables**: Height, Mass, Species, and Character Names (synthetic).
-   **Characteristics**: 
    -   Species distribution is skewed towards 'Human' (approx. 55%).
    -   Physical traits like height and mass are statistically distinct between species (e.g., Wookiees are taller/heavier, Hutts are very heavy).
-   **Purpose**: Ideal for data manipulation tasks (filtering, grouping), joining tables, and categorical visualization.
-   **Reproducibility**: Uses a fixed seed (default 42).

---

## Parameters

| Parameter | Type | Description | Default |
| :--- | :--- | :-------------------------------------------------------------------------- | :--- |
| `n`       | int  | Number of characters to generate.                                           | `87` |
| `seed`    | int  | Random seed for reproducibility.                                            | `42` |

---

## Returns

| Return      | Type          | Description                                                                 |
| :---------- | :------------ | :-------------------------------------------------------------------------- |
| `height_cm` | numpy.ndarray | Character heights in cm. Shape `(n,)`.                                      |
| `mass_kg`   | numpy.ndarray | Character masses in kg. Shape `(n,)`.                                       |
| `species`   | list[str]     | Species label for each entry (e.g., `'Human'`, `'Wookiee'`, `'Droid'`).     |
| `names`     | list[str]     | List of placeholder character names (e.g., `'Character 1'`).                |

---

## Example Usage

```python
from machinegnostics.data import make_starwars_check_data
import pandas as pd

# Generate character data
h, m, s, names = make_starwars_check_data()

# Create a DataFrame for easy viewing
df = pd.DataFrame({
    'Name': names,
    'Species': s,
    'Height': h,
    'Mass': m
})

print(df.head())
# Output (approx):
#           Name Species      Height       Mass
# 0  Character 1   Human  176.452312  81.231231
# 1  Character 2  Droid   168.123123  84.512341
# ...

# Find the average mass of Humans
human_mass = df[df['Species'] == 'Human']['Mass'].mean()
print(f"Avg Human Mass: {human_mass:.2f} kg")
```
