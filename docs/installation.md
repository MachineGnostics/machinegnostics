# Installation Guide

ManGo is distributed as a standard Python package and is designed for easy installation and integration into your data science workflow. The library has been tested on macOS with Python 3.11 and is fully compatible with standard data science libraries such as NumPy, pandas, and SciPy.

---

## 1. Create a Python Virtual Environment

It is best practice to use a virtual environment to manage your project dependencies and avoid conflicts with other Python packages.

```bash
# Create a new virtual environment named 'mango-env'
python3 -m venv mango-env

# Activate the environment (macOS/Linux)
source mango-env/bin/activate

# (On Windows, use: mango-env\Scripts\activate)
```

---

## 2. Install ManGo

Install the ManGo library using pip:

```bash
pip install mango
```

This command will install ManGo and automatically resolve its dependencies.

---

## 3. (Optional) Install Standard Data Science Libraries

If you do not already have them, install the most common data science libraries:

```bash
pip install numpy pandas scipy
```

---

## 4. Verify Installation

You can verify that ManGo and its dependencies are installed correctly by importing them in a Python session:

```python
import mango
import numpy
import pandas
import scipy

print("All libraries imported successfully!")
```

---

## 5. Quick Usage Example

ManGo is designed to be as simple to use as other machine learning libraries. You can call its functions and classes directly after installation.

```python
import mango as mg
import numpy as np
from mango.models import RobustRegressor

# Example data
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])

# Create and fit a robust polynomial regression model
model = RobustRegressor(degree=1)
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

print("Predictions:", y_pred)
```

---

## 6. Platform and Environment

- **Operating System:** Tested on macOS (Apple Silicon and Intel)
- **Python Version:** 3.11 recommended
- **Dependencies:** Compatible with NumPy, pandas, SciPy, and other standard data science libraries

---

## 7. Troubleshooting

- Ensure your virtual environment is activated before installing or running ManGo.
- If you encounter issues, try upgrading pip:
  ```bash
  pip install --upgrade pip
  ```
- For further help, consult the [official documentation](https://mango-gnostics.readthedocs.io/) or open an issue on the GitHub repository.

---

ManGo is designed for simplicity and reliability, making robust machine learning accessible for all Python users.
