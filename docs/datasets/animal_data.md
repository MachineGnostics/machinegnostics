# make_animals_check_data: Gnostic Animals Dataset

The `make_animals_check_data` function retrieves the 'Animals' dataset (Rousseeuw & Leroy, 1987), a classic small and challenging dataset for robust regression. It contains body weight and brain weight for 28 animals.

---

## Overview

This dataset is famous in robust statistics because it contains severe outliers that can skew traditional regression models. The goal is typically to model the relationship between body weight and brain weight.

-   **Features ($X$)**: Body weight in kilograms (kg).
-   **Target ($y$)**: Brain weight in grams (g).
-   **Challenge**: The dataset spans several orders of magnitude (from a Mouse to a Brachiosaurus). It includes 3 dinosaurs (Diplodocus, Brachiosaurus, Triceratops) which act as "bad leverage" points—they have massive body weights but relatively small brains compared to the mammalian trend.
-   **Suggested Usage**: Regression analysis, often with log-transformation: $\log(\text{Brain}) \sim \log(\text{Body})$.

---

## Returns

| Return  | Type           | Description                                                        |
| :------ | :------------- | :----------------------------------------------------------------- |
| `X`     | numpy.ndarray  | Body weight in kg. Shape `(28, 1)`.                                |
| `y`     | numpy.ndarray  | Brain weight in g. Shape `(28,)`.                                  |
| `names` | list of str    | The common name of the animal for each data point (e.g., "Mouse"). |

---

## Example Usage

```python
from machinegnostics.data import make_animals_check_data
import numpy as np

# Load the dataset
X, y, names = make_animals_check_data()
```

---

## Dataset Content

The dataset includes the following animals:
Mountain beaver, Cow, Grey wolf, Goat, Guinea pig, Diplodocus, Asian elephant, Rhesus monkey, Kangaroo, Golden hamster, Mouse, Rabbit, Sheep, Jaguar, Chimpanzee, Rat, Brachiosaurus, Mole, Pig, Echidna, Triceratops, Pigmy marmoset, African elephant, Human, Potar monkey, Cat, Giraffe, Gorilla.

---

**Source:** Rousseeuw, P. J., & Leroy, A. M. (1987). Robust Regression and Outlier Detection. John Wiley & Sons.

**Author:** Nirmal Parmar
