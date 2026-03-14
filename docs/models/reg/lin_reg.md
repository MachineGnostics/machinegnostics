# LinearRegressor: Robust Linear Regression with Machine Gnostics

The `LinearRegressor` is a robust linear regression model built on the Machine Gnostics framework. Unlike traditional statistical models that rely on probabilistic assumptions, this model uses algebraic and geometric structures to provide deterministic, resilient, and interpretable regression for real-world data.

---

## Overview

The Machine Gnostics LinearRegressor is designed for robust regression tasks, especially where data may contain outliers, noise, or non-Gaussian distributions. It leverages the core principles of Mathematical Gnostics (MG) to deliver reliable results even in challenging scenarios.

- **Deterministic & Finite:** No randomness or probability; all computations are reproducible.
- **Event-Level Modeling:** Handles uncertainty and error at the level of individual data events.
- **Algebraic Inference:** Utilizes gnostic algebra and error geometry for robust learning.
- **Resilient:** Designed to be robust against outliers, corrupted data, and distributional shifts.
- **Flexible:** Supports NumPy arrays and Pandas DataFrame/Series.
- **mlflow Integration:** For experiment tracking and deployment.
- **Easy Model Persistence:** Save and load models with joblib.

---

## Key Features

- Fits a linear regression model
- Robust to outliers and non-Gaussian noise
- Iterative optimization with early stopping and convergence tolerance
- Adaptive sample weighting using gnostic loss
- Training history tracking for analysis and visualization
- Customizable loss functions and scaling strategies
- Compatible with NumPy arrays and Pandas DataFrame/Series for input/output

---

## Parameters

| Parameter                 | Type                 | Default | Description                                                        |
| ------------------------- | -------------------- | ------- | ------------------------------------------------------------------ |
| `scale`                   | {'auto', int, float} | 'auto'  | Scaling method or value for input features.                        |
| `max_iter`                | int                  | 100     | Maximum number of optimization iterations.                         |
| `tolerance`               | float                | 1e-2    | Tolerance for convergence.                                         |
| `mg_loss`                 | str                  | 'hi'    | Gnostic loss function to use (`'hi'`, `'fi'`, etc.).               |
| `early_stopping`          | bool                 | True    | Whether to stop early if convergence is detected.                  |
| `verbose`                 | bool                 | False   | If True, prints progress and diagnostics during fitting.           |
| `data_form`               | str                  | 'a'     | Internal data representation format.                               |
| `gnostic_characteristics` | bool                 | False   | If True, computes and records gnostic properties (fi, hi, etc.).   |
| `history`                 | bool                 | True    | If True, records the optimization history for analysis.            |

---

## Attributes

- **coefficients**: `np.ndarray`
  Fitted linear regression coefficients.
- **weights**: `np.ndarray`
  Final sample weights after robust fitting.
- **params**: `list of dict`
  Parameter snapshots (loss, weights, gnostic properties) at each iteration.
- **_history**: `list` or `None`
  Internal optimization history (if enabled).
- **degree, max_iter, tolerance, mg_loss, early_stopping, verbose, scale, data_form, gnostic_characteristics**:
  Configuration parameters as set at initialization.

---

## Methods

### `fit(X, y)`

Fits the linear regressor to input features `X` and targets `y` using robust, gnostic loss minimization. Iteratively optimizes coefficients and sample weights, optionally recording history.

- **X**: array-like or DataFrame of shape `(n_samples, n_features)`
  Input features. Accepts NumPy arrays or Pandas DataFrame.
- **y**: array-like or Series of shape `(n_samples,)`
  Target values. Accepts NumPy arrays or Pandas Series/DataFrame column.

**Returns:**
`self` (fitted model instance)

---

### `predict(X)`

Predicts target values for new input features using the fitted model.

- **X**: array-like or DataFrame of shape `(n_samples, n_features)`
  Input features for prediction. Accepts NumPy arrays or Pandas DataFrame.

**Returns:**
`y_pred`: `np.ndarray`, shape `(n_samples,)`
Predicted target values.

---

### `score(X, y, case='i')`

Computes the robust (gnostic) R² score for the linear regressor model.

- **X**: array-like or DataFrame of shape `(n_samples, n_features)`
  Input features for scoring. Accepts NumPy arrays or Pandas DataFrame.
- **y**: array-like or Series of shape `(n_samples,)`
  True target values. Accepts NumPy arrays or Pandas Series/DataFrame column.
- **case**: `str`, default `'i'`
  Specifies the case or variant of the R² score to compute.

**Returns:**
`score`: `float`
Robust R² score of the model on the provided data.

---

### `save(path)`

Saves the trained model to disk using joblib.

- **path**: str
  Directory path to save the model.

---

### `load(path)`

Loads a previously saved model from disk.

- **path**: str
  Directory path where the model is saved.

**Returns:**
Instance of `LinearRegressor` with loaded parameters.

---

## Example Usage

=== "Python"

    ```python
    import numpy as np
    from machinegnostics.models import LinearRegressor
    import matplotlib.pyplot as plt
    
    # Generate linear data with noise and outliers
    np.random.seed(42)
    X = np.linspace(0, 2, 10).reshape(-1, 1)
    y = 10 * X.ravel() + 1  # True: y = 10x + 1
    
    # Add noise and an outlier
    y += np.random.normal(0, 2, 10)
    y[2] += 28.0  # Outlier
    
    # Fit the model
    model = LinearRegressor(max_iter=100, verbose=True)
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    print("Model fitted successfully!")
    print(f"Coefficients (Intercept, Slope): {model.coefficients}")
    print(f"MSE: {np.mean((y_pred - y)**2):.4f}")
    
    # Visualization code omitted for brevity; see output image
    ```

=== "Plot"

    ```python
    import matplotlib.pyplot as plt

    # Generate smooth line for visualization
    X_test = np.linspace(0, 2, 100).reshape(-1, 1)
    y_test = model.predict(X_test)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', s=100, label='Data', zorder=3, alpha=0.6)
    plt.scatter(X[2], y[2], color='red', s=150, label='Outlier', zorder=4, marker='X')
    plt.plot(X_test, y_test, 'g-', linewidth=2, label='Fitted Model')

    # True relationship for reference
    y_true = 10 * X_test + 1
    plt.plot(X_test, y_true, 'k--', linewidth=1, alpha=0.5, label='True: y=10x+1')

    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression Fit')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    ```

=== "Output"

    ![Linear Regression Fit](image/lin_reg/1770040477695.png)

---

## Training History

If `history=True`, the model records detailed training history at each iteration, accessible via `model.params` and `model._history`. Each entry contains:

- `iteration`: Iteration number
- `loss`: Gnostic loss value
- `coefficients`: Regression coefficients at this iteration (normalized)
- `rentropy`: Rentropy value (residual entropy)
- `weights`: Sample weights at this iteration
- `gnostic_characteristics`: (if enabled) fi, hi, etc.

This enables in-depth analysis and visualization of the training process.

---

## Notes

- The model is robust to outliers and suitable for datasets with non-Gaussian noise.
- Supports integration with mlflow for experiment tracking and deployment.

---

**Author:** Nirmal Parmar  
