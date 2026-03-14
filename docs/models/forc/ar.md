# AutoRegressor: Robust Time Series Forecasting

The `AutoRegressor` model implements an Autoregressive (AR) process for time series forecasting, enhanced by Mathematical Gnostics. Unlike standard AR models which often use Ordinary Least Squares (OLS), this model employs **Iterative Gnostic Reweighting**. This allows the model to autonomously down-weight outliers and anomalies in the historical data, resulting in a forecast that is robust to non-Gaussian noise and transient disruptions.

---

## Overview

Machine Gnostics `AutoRegressor` models the next value in a time series as a linear combination of its previous values (lags).

$$
y_t = c + w_1 y_{t-1} + w_2 y_{t-2} + \dots + w_p y_{t-p} + \epsilon_t
$$

The key innovation is the estimation of the coefficients ($w$). Instead of minimizing squared error (which is sensitive to outliers), it minimizes **Gnostic Entropy** via an iteratively reweighted least squares approach.

- **Robust Forecasting:** Resilient to outliers and structural breaks in history.
- **Trend Support:** Can model constant (bias) and linear time trends.
- **Recursive Forecasting:** Supports multi-step ahead prediction.
- **Iterative Refinement:** Optimizes weights to ignore noisy historical points.

---

## Key Features

- Robust Autoregressive (AR) modeling
- Outlier-resilient coefficient estimation
- Configurable lag order ($p$)
- Supports Constant ('c') and Constant+Linear ('ct') trends
- Recursive multi-step forecasting
- Gnostic diagnostics (weights, entropy)

---

## Parameters

| Parameter          | Type                 | Default    | Description                                                                                 |
| :----------------- | :------------------- | :--------- | :------------------------------------------------------------------------------------------ |
| `lags`           | `int`              | `1`      | Number of past observations to use ($p$).                                                 |
| `trend`          | `str`              | `'c'`    | Trend type:`'c'` (constant/bias), `'ct'` (constant + linear trend), `'n'` (no trend). |
| `scale`          | `str` \| `float` | `'auto'` | Scaling method or value for gnostic calculations.                                           |
| `max_iter`       | `int`              | `100`    | Maximum reweighting iterations.                                                             |
| `tolerance`      | `float`            | `1e-3`   | Convergence tolerance.                                                                      |
| `learning_rate`  | `float`            | `0.1`    | Learning rate for weight updates.                                                           |
| `mg_loss`        | `str`              | `'hi'`   | Gnostic loss function ('hi' or 'hj').                                                       |
| `early_stopping` | `bool`             | `True`   | Stop early if converged.                                                                    |
| `verbose`        | `bool`             | `False`  | Print progress.                                                                             |
| `history`        | `bool`             | `True`   | Record training history.                                                                    |

---

## Attributes

- **weights**: `np.ndarray`
  - The final robust weights assigned to the training samples. Low weights indicate potential outliers in the history.
- **coefficients**: `np.ndarray`
  - The fitted autoregressive coefficients (including trend parameters).
- **training_data_**: `np.ndarray`
  - Stored history used for recursive forecasting.
- **_history**: `list`
  - Training process history (loss, entropy) per iteration.

---

## Methods

### `fit(y, X=None)`

Fit the Autoregressor to the time series `y`.

**Parameters**

- **y**: array-like or Series
  - Target time series. Accepts NumPy arrays or Pandas Series.
- **X**: `Ignored`
  - Not used.

**Returns**

- **self**: `AutoRegressor`

---

### `predict(steps=1)`

Forecast future values recursively.

**Parameters**

- **steps**: `int`
  - Number of time steps to forecast into the future.

**Returns**

- **forecast**: `np.ndarray`
  - Predicted values for the next `steps`.

---

### `score(y, X=None)`

Evaluate the model on a time series `y` using **Robust R²**.
This reconstructs the lag features for `y` and calculates the one-step-ahead prediction accuracy.

**Parameters**

- **y**: array-like or Series
  - Time series to evaluate. Accepts NumPy arrays or Pandas Series.

**Returns**

- **score**: `float`
  - Robust R² score.

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

Instance of `AutoRegressor` with loaded parameters.

---

## Example Usage

=== "Python"

    ```python
    import numpy as np
    from machinegnostics.models import AutoRegressor
    import matplotlib.pyplot as plt

    # Generate synthetic AR data with outliers
    np.random.seed(42)
    t = np.arange(100)
    # Sine wave + Noise
    y = np.sin(t * 0.2) + np.random.normal(0, 0.1, 100)
    # Add spikes (outliers)
    y[20] = 5.0
    y[50] = -5.0
    y[80] = 5.0

    # Initialize AR(10) with constant trend
    model = AutoRegressor(lags=10, trend='c', verbose=True)

    # Fit
    model.fit(y)

    # Forecast next 20 steps
    future_steps = 20
    forecast = model.predict(steps=future_steps)

    print("Forecast:", forecast)

    # Visualization
    plt.plot(np.arange(100), y, label='History')
    plt.plot(np.arange(100, 100+future_steps), forecast, label='Forecast', color='red')
    plt.legend()
    plt.show()
    ```

=== "Example Output"

    ![Auto Regressor](image/ar/1770046639669.png)

---

## Notes

- **Lag Generation:** The model automatically creates the lag matrix $X$ where rows are windows $[y_{t-1}, \dots, y_{t-p}]$ and targets are $y_t$.
- **Trend Handling:**
  - `'c'`: Adds a column of 1s (bias).
  - `'ct'`: Adds a column of 1s and a column of time indices $t$.
- **Scaling:** Input data is scaled automatically (default) to ensure stable optimization of gnostic weights.

---

**Author:** Nirmal Parmar
