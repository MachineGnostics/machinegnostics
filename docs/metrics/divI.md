# divI: Divergence Information (DivI) Metric

The `divI` function computes the Divergence Information (DivI) metric, a robust measure for evaluating the divergence between observed data and model predictions. This metric is based on gnostic characteristics and is particularly useful for assessing the quality of model fits, especially in the presence of noise or outliers.

---

## Overview

Divergence Information (DivI) quantifies how much the information content of the predicted values diverges from that of the true values. Unlike classical divergence measures, DivI leverages gnostic algebra, making it robust to irregularities and non-Gaussian data.

Mathematically, DivI is defined as:

$$
\text{DivI} = \frac{1}{N} \sum_{i=1}^N \frac{I(y_i)}{I(\hat{y}_i)}
$$

where:

- $I(y_i)$ is the E-information of the observed value $y_i$,
- $I(\hat{y}_i)$ is the E-information of the fitted value $\hat{y}_i$,
- $N$ is the number of data points.

DivI compares the information content of the dependent variable and its fit. The better the fit, the closer DivI is to 1. If the fit is highly uncertain or poor, DivI decreases.

- Compatibility: Accepts NumPy arrays and Pandas Series for inputs.

---

## Interpretation

- **Higher DivI**: Indicates that the fitted values retain more of the information content of the observed data, suggesting a better model fit.
- **Lower DivI**: Indicates greater divergence between the distributions of the observed and fitted values, suggesting a poorer fit or higher uncertainty in the model.

DivI is particularly useful in robust model evaluation, as it is less sensitive to outliers and non-normal data distributions.

---

## Parameters

| Parameter   | Type                           | Description                                                     |
| ----------- | ------------------------------ | --------------------------------------------------------------- |
| `y`       | array-like or Pandas Series     | Observed data (ground truth). 1D values.                        |
| `y_fit`   | array-like or Pandas Series     | Fitted data (model predictions). 1D values, same shape as `y`. |
| `verbose` | bool                            | Print detailed progress, warnings, and results                  |

---

## Returns

- **float**
  The computed Divergence Information (DivI) value.

---

## Raises

- **ValueError**
  - If `y` and `y_fit` do not have the same shape.
  - If `y` or `y_fit` are not 1D arrays.

---

## Example Usage

```python
import numpy as np
from machinegnostics.metrics import divI

y = np.array([1.0, 2.0, 3.0, 4.0])
y_fit = np.array([1.1, 1.9, 3.2, 3.8])
result = divI(y, y_fit)
print(result)
```

---

## Notes

- DivI is calculated using gnostic characteristics, providing a robust way to measure divergence between distributions.
- The metric is especially useful for model evaluation in real-world scenarios where data may be noisy or contain outliers.
- In the context of model evaluation, DivI is often used alongside other criteria such as Robust R-squared (RobR2) and the Geometric Mean of Multiplicative Fitting Errors (GMMFE) to provide a comprehensive assessment of model performance.

---

**Author:** Nirmal Parmar	
**Date:** 2025-09-24

---
