# SARIMA: Robust Seasonal ARIMA

The `SARIMA` model (Seasonal AutoRegressive Integrated Moving Average) extends the robust forecasting capabilities of Gnostic ARIMA to data with seasonal patterns. It supports both non-seasonal $(p,d,q)$ and seasonal $(P,D,Q,s)$ components, estimated using robust Gnostic Weighted Least Squares.

---

## Overview

Machine Gnostics `SARIMA` allows for robust modeling of complex time series that exhibit periodic behavior (seasonality), such as monthly sales or daily temperatures, while maintaining resilience against outliers.

- **Seasonal Integrated (D):** Handles seasonal non-stationarity (e.g., year-over-year growth).
- **Seasonal AR/MA (P, Q):** Models relationships at seasonal lags.
- **Robust Estimation:** Down-weights outliers that might distort seasonal pattern detection.
- **Recursive Forecasting:** Supports multi-step ahead prediction by unrolling the differencing layers.

---

## Key Features

- Robust SARIMA(p,d,q)x(P,D,Q,s) modeling
- Automatic handling of seasonal differencing
- Supports Constant ('c') and Constant+Linear ('ct') trends
- Robust to outliers via Gnostic Weights
- Recursive multi-step forecasting

---

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `order` | `tuple` | `(1, 0, 0)` | The non-seasonal $(p, d, q)$ order. |
| `seasonal_order` | `tuple` | `(0, 0, 0, 0)` | The seasonal $(P, D, Q, s)$ order. $s$ is the periodicity. |
| `trend` | `str` | `'c'` | Trend type: `'c'` (bias), `'ct'` (linear), `'n'` (none). |
| `optimize` | `bool` | `False` | (Experimental) Auto-select orders. |
| `scale` | `str` \| `float` | `'auto'` | Scaling method for gnostic calculations. |
| `max_iter` | `int` | `100` | Maximum reweighting iterations. |
| `tolerance` | `float` | `1e-3` | Convergence tolerance. |
| `learning_rate` | `float` | `0.1` | Learning rate for weight updates. |
| `mg_loss` | `str` | `'hi'` | Gnostic loss function ('hi' or 'hj'). |
| `early_stopping` | `bool` | `True` | Stop early if converged. |
| `history` | `bool` | `True` | Record training history. |

---

## Attributes

- **weights**: `np.ndarray`
    - The final robust weights assigned to the training samples.
- **coefficients**: `np.ndarray`
    - The fitted model parameters.
- **training_data_diff_**: `np.ndarray`
    - The fully stationarized series (after both differences) used internally.
- **training_residuals_**: `np.ndarray`
    - Estimated residuals from the training phase.

---

## Methods

### `fit(y, X=None)`

Fit the SARIMA model to the time series `y`.

**Parameters**

- **y**: array-like or Series
    - Target time series. Accepts NumPy arrays or Pandas Series.
- **X**: `Ignored`
    - Not used.

**Returns**

- **self**: `SARIMA`

---

### `predict(steps=1)`

Forecast future values recursively.

This method handles the complex inverse transformation of both regular ($d$) and seasonal ($D$) differencing to return forecasts on the original scale.

**Parameters**

- **steps**: `int`
    - Number of time steps to forecast.

**Returns**

- **forecast**: `np.ndarray`
    - Predicted values for the next `steps`.

---

### `score(y, X=None)`

Evaluate the model on the training time series `y`.
Returns the **Robust R²** calculated on the stationary (fully differenced) series.

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

Instance of `SARIMA` with loaded parameters.

---

## Example Usage

=== "Python"

    ```python
    import numpy as np
    from machinegnostics.models import SARIMA
    import matplotlib.pyplot as plt

    # Generate synthetic Seasonal Data (Period=12)
    np.random.seed(42)
    t = np.arange(120)
    seasonal = np.sin(2 * np.pi * t / 12)
    trend = 0.05 * t
    noise = np.random.normal(0, 0.2, 120)
    y = seasonal + trend + noise
    
    # Add seasonal outlier
    y[13] += 5.0 # Spike in month 2 of year 2

    # Initialize SARIMA(1,0,0)x(1,1,0,12)
    # Simple AR(1) with Seasonal Integration and Seasonal AR(1)
    model = SARIMA(
        order=(1, 0, 0),
        seasonal_order=(1, 1, 0, 12),
        trend='c',
        verbose=True
    )
    
    # Fit
    model.fit(y)
    
    # Forecast next 24 steps (2 seasons)
    forecast = model.predict(steps=24)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(120), y, label='History')
    plt.plot(np.arange(120, 144), forecast, label='Forecast', color='red')
    plt.title("Gnostic SARIMA Forecast")
    plt.legend()
    plt.show()
    ```

=== "Example Output"

    ![SARIMA](image/sarima/1770047099472.png)

---

## Notes

- **Initial Residuals:** For MA terms, initial residuals are estimated using a high-order AR model covering the seasonal period.
- **Differencing:** The model applies seasonal differencing ($D$) first, followed by regular differencing ($d$). Forecasts reverse this order.
- **Complexity:** Higher seasonal orders (large $P, Q, s$) significantly increase the number of parameters and required history length.

---

**Author:** Nirmal Parmar  
