# ARIMA: Robust AutoRegressive Integrated Moving Average

The `ARIMA` model (AutoRegressive Integrated Moving Average) extends robust time series forecasting to non-stationary data and moving average processes. Enhanced by Mathematical Gnostics, this implementation uses iterative reweighting to minimize the influence of outliers on the estimation of $p$, $d$, and $q$ parameters.

---

## Overview

Machine Gnostics `ARIMA` brings robustness to the classic ARIMA framework. Standard estimation methods (like MLE) are highly sensitive to outliers. This implementation uses **Gnostic Weighted Least Squares** to robustly estimate autoregressive and moving average parameters.

- **Integrated (d):** Automatically handles differencing to stationarize data.
- **AutoRegressive (p):** Models relationships with past values.
- **Moving Average (q):** Models relationships with past forecast errors.
- **Auto-Optimization:** Can search for the optimal $(p, d, q)$ order.
- **Robustness:** Down-weights anomalous time steps during training.

---

## Key Features

- Robust ARIMA(p,d,q) modeling
- Automatic order selection (Grid Search)
- Handles non-stationary data via differencing
- Supports Constant ('c') and Constant+Linear ('ct') trends
- Recursive multi-step forecasting
- Gnostic diagnostics (weights, entropy)

---

## Parameters

| Parameter            | Type                 | Default       | Description                                                      |
| :------------------- | :------------------- | :------------ | :--------------------------------------------------------------- |
| `order`            | `tuple`            | `(1, 0, 0)` | The$(p, d, q)$ order of the model.                             |
| `optimize`         | `bool`             | `False`     | If True, searches for optimal order within `max_order_search`. |
| `max_order_search` | `tuple`            | `(5, 1, 5)` | Max$(p, d, q)$ limits for optimization.                        |
| `trend`            | `str`              | `'c'`       | Trend type:`'c'` (bias), `'ct'` (linear), `'n'` (none).    |
| `scale`            | `str` \| `float` | `'auto'`    | Scaling method for gnostic calculations.                         |
| `max_iter`         | `int`              | `100`       | Maximum reweighting iterations.                                  |
| `tolerance`        | `float`            | `1e-3`      | Convergence tolerance.                                           |
| `learning_rate`    | `float`            | `0.1`       | Learning rate for weight updates.                                |
| `mg_loss`          | `str`              | `'hi'`      | Gnostic loss function ('hi' or 'hj').                            |
| `early_stopping`   | `bool`             | `True`      | Stop early if converged.                                         |
| `verbose`          | `bool`             | `False`     | Print progress.                                                  |
| `history`          | `bool`             | `True`      | Record training history.                                         |

---

## Attributes

- **weights**: `np.ndarray`
  - The final robust weights assigned to the training samples.
- **coefficients**: `np.ndarray`
  - The fitted model parameters ($\phi$, $\theta$, trend).
- **p, d, q**: `int`
  - The final model order (updated if `optimize=True`).
- **training_data_raw_**: `np.ndarray`
  - Original training series (used for inverse differencing during forecast).
- **training_residuals_**: `np.ndarray`
  - Estimated residuals from the training phase.

---

## Methods

### `fit(y, X=None)`

Fit the ARIMA model to the time series `y`.

If `optimize=True`, this runs a grid search over $(p, d, q)$ combinations to minimize RMSE on a validation split before fitting the final model.

**Parameters**

- **y**: array-like or Series
  - Target time series. Accepts NumPy arrays or Pandas Series.
- **X**: `Ignored`
  - Not used.

**Returns**

- **self**: `ARIMA`

---

### `predict(steps=1)`

Forecast future values recursively.

This method handles:

1. Recursive prediction of differenced values.
2. Inverse differencing to return forecasts on the original scale.

**Parameters**

- **steps**: `int`
  - Number of time steps to forecast.

**Returns**

- **forecast**: `np.ndarray`
  - Predicted values for the next `steps`.

---

### `score(y, X=None)`

Evaluate the model on a time series `y` using **Robust R²**.
Currently, scoring is primarily supported for in-sample evaluation (on the training data) due to the recursive nature of MA terms.

**Parameters**

- **y**: array-like or Series
  - Time series to evaluate. Accepts NumPy arrays or Pandas Series.

**Returns**

- **score**: `float`
  - Robust R² score (on the stationary/differenced series).

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

**Returns**

Instance of `ARIMA` with loaded parameters.

---

## Example Usage

=== "Python"

    ```python
    import numpy as np
    from machinegnostics.models import ARIMA
    import matplotlib.pyplot as plt

    # Generate synthetic ARIMA(1,1,1) process
    np.random.seed(42)
    n = 150
    # Random walk with drift
    y = np.cumsum(np.random.normal(0.1, 1, n))

    # Add outliers
    y[50] += 10
    y[100] -= 10

    # Initialize model with automic order search
    model = ARIMA(
        optimize=True,
        max_order_search=(3, 2, 3),
        trend='c',
        verbose=True
    )

    # Fit (will search for best p,d,q)
    model.fit(y)

    print(f"Best Order Found: ({model.p}, {model.d}, {model.q})")

    # Forecast
    forecast = model.predict(steps=20)

    # Plot
    plt.plot(np.arange(n), y, label='History')
    plt.plot(np.arange(n, n+20), forecast, label='Forecast', color='red')
    plt.legend()
    plt.show()
    ```

=== "Example Output"

    ![ARIMA](image/arima/1770046880571.png)

---

## Notes

- **Initial Residuals:** For MA terms ($q > 0$), initial residuals are estimated using the Hannan-Rissanen algorithm (approximated via a long AR model).
- **Optimization:** The order search evaluates models based on RMSE on the last 20% of the data.
- **Differencing:** The internal states store the differenced series. Forecasts are integrated back to original scale automatically.

---

**Author:** Nirmal Parmar
