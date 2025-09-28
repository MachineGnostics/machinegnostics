# Machine Gnostics

> Machine Gnostics: Laws of Nature, Encoded—For Everyone!

Welcome to **Machine Gnostics**, an innovative Python library designed to implement the principles of **Machine Gnostics** for robust data analysis, modeling, and inference. Unlike traditional statistical approaches that depend heavily on probabilistic assumptions, Machine Gnostics harnesses deterministic algebraic and geometric structures. This unique foundation enables the library to deliver exceptional resilience against outliers, noise, and corrupted data, making it a powerful tool for challenging real-world scenarios.

Machine Gnostics is an open-source initiative that seeks to redefine the mathematical underpinnings of machine learning. While most conventional ML libraries are grounded in probabilistic and statistical frameworks, Machine Gnostics explores alternative paradigms—drawing from deterministic algebra, information theory, and geometric methods. This approach opens new avenues for building robust, interpretable, and reliable analysis tools that can withstand the limitations of traditional models.

> As a pioneering project, Machine Gnostics invites users to adopt a fresh perspective and develop a new understanding of machine learning. The library is currently in its infancy, and as such, some features may require refinement and fixes. We are actively working to expand its capabilities, with new models and methods planned for the near future. Community support and collaboration are essential to realizing Machine Gnostics’ full potential. Together, let’s build a new AI grounded in a rational and resilient paradigm.

More information at [Machine Gnostics](https://machinegnostics.info)

---

# Installation Guide

Machine Gnostics is distributed as a standard Python package and is designed for easy installation and integration into your data science workflow. The library has been tested on macOS with Python 3.11 and is fully compatible with standard data science libraries.

---

## 1. Create a Python Virtual Environment

It is best practice to use a virtual environment to manage your project dependencies and avoid conflicts with other Python packages.

### macOS & Linux

```bash
# Create a new virtual environment named 'mg-env'
python3 -m venv mg-env
# Activate the environment
source mg-env/bin/activate
```

### Windows

```cmd
# Create a new virtual environment named 'mg-env'
python -m venv mg-env
# Activate the environment
mg-env\Scripts\activate
```

---

## 2. Install Machine Gnostics

Install the Machine Gnostics library using pip:

### macOS & Linux

```bash
pip install machinegnostics
```

### Windows

```cmd
pip install machinegnostics
```

This command will install Machine Gnostics and automatically resolve its dependencies.

---

## 3. Verify Installation

You can verify that Machine Gnostics and its dependencies are installed correctly by importing them in a Python session:

```python
# check import
import machinegnostics
print("imported successfully!")
```

You can also check the installation with pip:

### macOS & Linux

```bash
pip show machinegnostics
```

### Windows

```cmd
pip show machinegnostics
```

---

## 4. Quick Usage Example

Machine Gnostics is designed to be as simple to use as other machine learning libraries. You can call its functions and classes directly after installation.

### Example "Gnostic Distribution Function"

```python
import numpy as np
from machinegnostics.magcal import EGDF

data = np.array([ -13.5, 0, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
egdf = EGDF()
egdf.fit(data)
egdf.plot()
print(egdf.params)
```

### Example "Polynomial Regression"

```python
import numpy as np
from machinegnostics.models.regression import PolynomialRegressor

# Example data
X = np.array([0., 0.4, 0.8, 1.2, 1.6, 2. ])
y = np.array([17.89408548, 69.61586934, -7.19890572, 9.37670866, -10.55673099, 16.57855348])

# Create and fit a robust polynomial regression model
model = PolynomialRegressor(degree=2)
model.fit(X, y)

model_lr = LinearRegressor()
model_lr.fit(X, y)

# Make predictions
y_pred = model.predict(X)
y_pred_lr = model_lr.predict(X)

print("Predictions:", y_pred)

# coefficients
print("Coefficients:", model.coefficients)

# x vs y, y_pred plot
import matplotlib.pyplot as plt
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, y_pred, color='red', label='Polynomial Prediction')
plt.plot(X, y_pred_lr, color='green', label='Linear Prediction')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 5. Platform and Environment

- **Operating System:** Tested on macOS and Windows 11
- **Python Version:** 3.11 recommended
- **Dependencies:** Compatible with NumPy, pandas, SciPy, and other standard data science libraries

---

## 6. Troubleshooting

- **Activate Your Environment:**
  Always activate your virtual environment before installing or running Machine Gnostics.

  **macOS & Linux:**

  ```bash
  source mg-env/bin/activate
  # or for conda
  conda activate myenv
  ```

  **Windows:**

  ```cmd
  mg-env\Scripts\activate
  # or for conda
  conda activate myenv
  ```
- **Check Your Python Version:**
  Ensure you are using Python 3.8 or newer.

  **macOS & Linux:**

  ```bash
  python3 --version
  ```

  **Windows:**

  ```cmd
  python --version
  ```
- **Upgrade pip:**
  An outdated pip can cause installation errors. Upgrade pip before installing:

  **macOS & Linux:**

  ```bash
  pip install --upgrade pip
  ```

  **Windows:**

  ```cmd
  pip install --upgrade pip
  ```

**Install from a Clean Environment:** If you encounter conflicts, try creating a fresh virtual environment and reinstalling.
**Check Your Internet Connection:** Download errors often result from network issues. Make sure you are connected.
**Permission Issues:** If you see permission errors, avoid using `sudo pip install`. Instead, use a virtual environment.

- **Still Stuck**

  - [Contact us](https://machinegnostics.info/contact/) or open an issue on [GitHub](https://github.com/MachineGnostics/machinegnostics).

---

Machine Gnostics is designed for simplicity and reliability, making robust machine learning accessible for all Python users.
