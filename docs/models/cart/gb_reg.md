# GnosticBoostingRegressor: Robust Boosting with Machine Gnostics

The `GnosticBoostingRegressor` extends the Gradient Boosting (XGBoost) approach by integrating Mathematical Gnostics principles. It employs an iterative reweighting scheme that assesses the quality of each data sample based on the residuals of the previous iteration's model. This allows the model to autonomously down-weight outliers and noise, resulting in a robust regression model capable of handling contaminated data.

This implementation wraps the XGBoost library, combining its high-performance gradient boosting with the robust statistical framework of Machine Gnostics.

---

## Overview

Machine Gnostics `GnosticBoostingRegressor` leverages the ensemble power of Gradient Boosting Trees along with the robust weighting of Machine Gnostics. It is particularly effective for regression tasks where the data may contain outliers or follow non-standard distributions.

- **Robustness to Outliers:** Automatically identifies and down-weights anomalous samples during training.
- **Iterative Refinement:** Optimizes sample weights over multiple iterations until convergence.
- **Boosted Performance:** Uses Gradient Boosting (XGBoost) as the underlying estimator for state-of-the-art performance.
- **Event-Level Modeling:** Handles uncertainty at the level of individual data events.
- **Easy Model Persistence:** Save and load models with joblib.

---

## Key Features

- Robust regression using iterative gnostic reweighting
- Iterative refinement of sample weights
- XGBoost hyperparameters (n_estimators, max_depth, learning_rate)
- Identifies and handles outliers automatically
- Convergence-based early stopping
 - Training history tracking for analysis
 - Compatible with NumPy arrays and Pandas DataFrame/Series for input/output

---

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `n_estimators` | `int` | `100` | The number of boosting rounds (trees). |
| `max_depth` | `int` | `6` | Maximum depth of the base learners. |
| `learning_rate` | `float` | `0.3` | Boosting learning rate (eta). |
| `gnostic_weights` | `bool` | `True` | Whether to use iterative gnostic weights. |
| `max_iter` | `int` | `10` | Maximum gnostic iterations. |
| `tolerance` | `float` | `1e-4` | Convergence tolerance. |
| `data_form` | `str` | `'a'` | Data form: 'a' (additive) or 'm' (multiplicative). |
| `verbose` | `bool` | `False` | Verbosity. |
| `random_state` | `int` | `None` | Random seed. |
| `history` | `bool` | `True` | Whether to record training history. |
| `scale` | `str` \| `float` | `'auto'` | Scaling method for input features. |
| `early_stopping` | `bool` | `True` | Whether to stop training early if convergence is detected. |
| `**kwargs` | `dict` | | Additional arguments passed to `xgboost.XGBRegressor`. |

---

## Attributes

- **weights**: `np.ndarray`
    - The final calibrated sample weights assigned to the training data.
- **model**: `xgboost.XGBRegressor`
    - The underlying fitted XGBoost model.
- **_history**: `list`
    - List of dictionaries containing training history (loss, entropy, weights).
- **n_estimators, gnostic_weights, max_depth, learning_rate**
    - Configuration parameters as set at initialization.

---

## Methods

### `fit(X, y)`

Fit the Gnostic Boosting model to the training data.

This method trains the boosting regressor. If `gnostic_weights` is True, it iteratively refines the model by reweighting samples based on gnostic residuals to down-weight outliers.

**Parameters**

 - **X**: array-like or DataFrame of shape `(n_samples, n_features)`
     - Input features. Accepts NumPy arrays or Pandas DataFrame.
 - **y**: array-like or Series of shape `(n_samples,)`
     - Target values. Accepts NumPy arrays or Pandas Series/DataFrame column.

**Returns**

- **self**: `GnosticBoostingRegressor`
    - Returns the fitted model instance for chaining.

---

### `predict(model_input)`

Predict target values for input samples using the boosted ensemble.

**Parameters**

 - **model_input**: array-like or DataFrame of shape `(n_samples, n_features)`
     - Input data for prediction. Accepts NumPy arrays or Pandas DataFrame.

**Returns**

- **y_pred**: `np.ndarray` of shape `(n_samples,)`
    - Predicted target values.

---

### `score(X, y)`

Return the robust (gnostic) coefficient of determination R² of the prediction.

**Parameters**

 - **X**: array-like or DataFrame of shape `(n_samples, n_features)`
     - Input features for evaluation. Accepts NumPy arrays or Pandas DataFrame.
 - **y**: array-like or Series of shape `(n_samples,)`
     - True target values. Accepts NumPy arrays or Pandas Series/DataFrame column.

**Returns**

- **score**: `float`
    - Robust R² score of the model.

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

Instance of `GnosticBoostingRegressor` with loaded parameters.

---

## Example Usage

=== "Python"

    ```python
    import numpy as np
    from machinegnostics.models import GnosticBoostingRegressor

    # Generate synthetic data
    np.random.seed(42)
    X = np.random.rand(100, 5)
    # Target is sum of features + noise
    y = np.sum(X, axis=1) + np.random.normal(0, 0.1, 100)
    
    # Add strong outliers
    y[::10] += 20  
    
    # Initialize and fit the robust boosting model
    model = GnosticBoostingRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        gnostic_weights=True,
        max_iter=5,
        verbose=True
    )
    model.fit(X, y)

    # Make predictions
    preds = model.predict(X[:5])
    print("Predictions:", preds)
    
    # Score
    r2 = model.score(X, y)
    print(f"Robust R2: {r2:.4f}")
    ```

=== "Example Output"

    ![Gnostic Boosting Reg](image/gb_reg/1770045418116.png)


---

## Training History

If `history=True`, the model records detailed training history at each iteration, accessible via `model._history`.  This helps in analyzing how the model identifies and down-weights noisy samples over time.

---

## Notes

- This model requires `xgboost` to be installed.
- It is particularly effective for datasets where outliers would otherwise skew the boosting process significantly.

---

**Author:** Nirmal Parmar  
