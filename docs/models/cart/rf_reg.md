# GnosticRandomForestRegressor: Robust Random Forest with Machine Gnostics

The `GnosticRandomForestRegressor` extends the standard random forest approach by integrating Mathematical Gnostics principles. It employs an iterative reweighting scheme that assesses the quality of each data sample based on the residuals of the previous iteration's model. This allows the forest to autonomously down-weight outliers and noise, resulting in a more robust predictive model.

---

## Overview

Machine Gnostics `GnosticRandomForestRegressor` combines the ensemble power of Random Forests with the robust weighting of Machine Gnostics. This makes it exceptionally capable in regression tasks where the training data contains outliers or follows non-Gaussian noise distributions.

- **Robustness to Outliers:** Automatically identifies and down-weights anomalous samples during training.
- **Iterative Refinement:** Optimizes sample weights over multiple iterations until convergence.
- **Ensemble Learning:** Leverages multiple decision trees for reduced variance and improved generalization.
- **Event-Level Modeling:** Handles uncertainty at the level of individual data events.
- **Easy Model Persistence:** Save and load models with joblib.

---

## Key Features

- Robust regression using iterative gnostic reweighting
- Iterative refinement of sample weights
- Standard Random Forest hyperparameters (n_estimators, max_depth, etc.)
- Identifies and handles outliers automatically
- Convergence-based early stopping
- Training history tracking for analysis
- Compatible with NumPy arrays and Pandas DataFrame/Series for input/output

---

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `n_estimators` | `int` | `100` | The number of trees in the forest. |
| `max_depth` | `int` | `None` | Maximum depth of the tree. |
| `min_samples_split` | `int` | `2` | Minimum samples to split a node. |
| `gnostic_weights` | `bool` | `True` | Whether to use iterative gnostic weights. |
| `max_iter` | `int` | `10` | Maximum gnostic iterations. |
| `tolerance` | `float` | `1e-4` | Convergence tolerance. |
| `data_form` | `str` | `'a'` | Data form: 'a' (additive) or 'm' (multiplicative). |
| `verbose` | `bool` | `False` | Verbosity. |
| `random_state` | `int` | `None` | Random seed. |
| `history` | `bool` | `True` | Whether to record training history. |
| `scale` | `str` \| `float` | `'auto'` | Scaling method for input features. |
| `early_stopping` | `bool` | `True` | Whether to stop training early if convergence is detected. |

---

## Attributes

- **weights**: `np.ndarray`
    - The final calibrated sample weights assigned to the training data.
- **trees**: `list`
    - The list of underlying regression trees (estimators) that make up the forest.
- **_history**: `list`
    - List of dictionaries containing training history (loss, entropy, weights).
- **n_estimators, gnostic_weights, max_depth, max_iter, tolerance**
    - Configuration parameters as set at initialization.

---

## Methods

### `fit(X, y)`

Fit the Gnostic Forest model to the training data.

This method trains the random forest classifier. If `gnostic_weights` is True, it iteratively refines the model by reweighting samples based on gnostic residuals to down-weight outliers.

**Parameters**

- **X**: array-like or DataFrame of shape `(n_samples, n_features)`
    - Input features. Accepts NumPy arrays or Pandas DataFrame.
- **y**: array-like or Series of shape `(n_samples,)`
    - Target values. Accepts NumPy arrays or Pandas Series/DataFrame column.

**Returns**

- **self**: `GnosticRandomForestRegressor`
    - Returns the fitted model instance for chaining.

---

### `predict(model_input)`

Predict target values for input samples.

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

Instance of `GnosticRandomForestRegressor` with loaded parameters.

---

## Example Usage

=== "Python"

    ```python
    import numpy as np
    from machinegnostics.models import GnosticRandomForestRegressor

    # Generate synthetic data with outliers
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    # True relation: y = 2x + 1
    y = 2 * X.ravel() + 1 + np.random.normal(0, 0.5, 100)
    # Add strong outliers
    y[::10] += 20  
    
    # Initialize and fit the robust forest model
    model = GnosticRandomForestRegressor(
        n_estimators=50,
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

    ![Gnostic Random Forest Reg](image/rf_reg/1770045211018.png)


---

## Training History

If `history=True`, the model records detailed training history at each iteration, accessible via `model._history`.  This helps in analyzing how the model identifies and down-weights noisy samples over time.

---

## Notes

- This model is particularly effective when the training data contains localized outliers or non-Gaussian noise.
- The `gnostic_weights` mechanism allows the forest to "self-clean" the data during training.

---

**Author:** Nirmal Parmar  
