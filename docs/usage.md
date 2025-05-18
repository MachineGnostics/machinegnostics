# Usage Guide

This guide provides a comprehensive overview of how to use the **ManGo** library for robust data analysis and machine learning based on Machine Gnostics principles. ManGo offers robust regression models, gnostic metrics, and alternative statistical tools designed to be resilient to outliers and corrupted data.

---

## 1. Importing ManGo

After installation, you can import ManGo and its modules in your Python scripts or notebooks:

```python
import mango as mg
from mango.models import RobustRegressor
from mango.metrics import robr2, gmmfe, divI, evalMet, hc
from mango.magcal import gmedian, gvariance, gautocovariance, gcorrelation, gcovariance
```

---

## 2. Robust Regression

ManGo provides robust regression models that are less sensitive to outliers.

**Example: Using `RobustRegressor`**

```python
from mango.models import RobustRegressor

# X: feature matrix, y: target vector
model = RobustRegressor()
model.fit(X, y)
y_pred = model.predict(X_test)
```

---

## 3. Gnostic Metrics

Evaluate your models with robust, gnostic metrics:

```python
from mango.metrics import robr2, gmmfe

score = robr2(y_true, y_pred)
gmmfe = gmmfe(y_true, y_pred)
hc = hc(y_true, y_pred, case='i') # estimating case
```

Other available metrics:

- `divI`: Divergence Index
- `evalMet`: General evaluation metric
- `hc`: Relavance of the given data samples

## 4. Gnostic Statistical Tools

ManGo includes robust alternatives to classical statistics:

```python
gmed = gmedian(data)
gmod = gmodulus(data)
gvar = gvariance(data)
gacov = gautocovariance(data1, data2)
gcor = gcorrelation(data1, data2)
gcov = gcovariance(data1, data2)
```

---

## 5. Example Workflow

```python
import numpy as np
from mango.models import RobustRegressor
from mango.metrics import robr2

# Generate synthetic data
X = np.random.randn(100, 3)
y = 2 * X[:, 0] - X[:, 1] + np.random.randn(100) * 0.5

# Fit robust regression
model = RobustRegressor()
model.fit(X, y)
y_pred = model.predict(X)

# Evaluate
score = robr2(y, y_pred)
print("Robust R2:", score)

```

---

## 6. Additional Resources

- [Examples](./examples.md)
- [Machine Gnostics Principles](https://github.com/your-org/ManGo/wiki)

---

## 7. Troubleshooting

- **ImportError**: Ensure ManGo is installed and your `PYTHONPATH` includes the `src` directory.
- **Unexpected Results**: Check for outliers or corrupted data in your input.

---

For further help, open an issue on [GitHub](https://github.com/your-org/ManGo/issues) or contact the maintainers.
