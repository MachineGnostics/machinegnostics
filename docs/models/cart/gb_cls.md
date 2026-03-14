# GnosticBoostingClassifier: Robust Gradient Boosting with Machine Gnostics

The `GnosticBoostingClassifier` integrates the power of Gradient Boosting (via XGBoost) with the robustness of Mathematical Gnostics. It employs an iterative reweighting mechanism that uses gnostic loss functions to assess data quality, allowing the model to autonomously down-weight outliers and noisy samples during training.

---

## Overview

Machine Gnostics `GnosticBoostingClassifier` combines state-of-the-art gradient boosting with rigorous error handling. By iteratively refining sample weights based on gnostic criteria (like rentropy and information loss), it achieves superior stability and accuracy in datasets with label noise or outliers.

- **Robust Boosting:** Upgrades standard XGBoost with gnostic error modeling.
- **Iterative Refinement:** Optimizes sample weights over multiple iterations to minimize gnostic loss.
- **Data Quality Handling:** Automatically identifies and down-weights low-fidelity or mislabeled samples.
- **Configurable:** Supports standard boosting parameters along with gnostic settings.
- **Event-Level Modeling:** Handles uncertainty at the level of individual data events.
- **Easy Model Persistence:** Save and load models with joblib.

---

## Key Features

- Robust classification using iterative gnostic reweighting
- Integration with XGBoost for high-performance gradient boosting
- Customizable gnostic loss functions (`'hi'`, etc.)
- Convergence-based early stopping
 - Training history tracking for detailed analysis
 - Compatible with NumPy arrays and Pandas DataFrame/Series for input/output

---

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `n_estimators` | `int` | `100` | Number of boosting rounds. |
| `max_depth` | `int` | `6` | Maximum tree depth for base learners. |
| `learning_rate` | `float` | `0.3` | Boosting learning rate. |
| `max_iter` | `int` | `10` | Maximum number of gnostic reweighting iterations. |
| `tolerance` | `float` | `1e-4` | Convergence tolerance for early stopping. |
| `mg_loss` | `str` | `'hi'` | Gnostic loss function to use. |
| `data_form` | `str` | `'a'` | Data form: 'a' (additive) or 'm' (multiplicative). |
| `verbose` | `bool` | `False` | Verbosity. |
| `random_state` | `int` | `None` | Random seed. |
| `history` | `bool` | `True` | Whether to record training history. |
| `scale` | `str` \| `float` | `'auto'` | Scaling method for input features. |
| `early_stopping` | `bool` | `True` | Whether to stop training early if convergence is detected. |
| `gnostic_characteristics`| `bool` | `False` | Whether to compute extended gnostic metrics. |

---

## Attributes

- **model**: `Any`
    - The underlying XGBoost classifier instance.
- **weights**: `np.ndarray`
    - The final calibrated sample weights.
- **classes_**: `np.ndarray`
    - Class labels.
- **_history**: `list`
    - List of dictionaries containing training history (loss, entropy, weights).
- **n_estimators, max_depth, learning_rate, max_iter, tolerance**
    - Configuration parameters as set at initialization.

---

## Methods

### `fit(X, y)`

Fit the Gnostic Boosting model to the training data.

This method trains the model using an iterative process. In each iteration, an XGBoost classifier is trained, predictions are made, and sample weights are updated based on the gnostic loss of the residuals. This process repeats until convergence or `max_iter`.

**Parameters**

 - **X**: array-like or DataFrame of shape `(n_samples, n_features)`
     - Input features. Accepts NumPy arrays or Pandas DataFrame.
 - **y**: array-like or Series of shape `(n_samples,)`
     - Target labels. Accepts NumPy arrays or Pandas Series/DataFrame column.

**Returns**

- **self**: `GnosticBoostingClassifier`
    - Returns the fitted model instance for chaining.

---

### `predict(model_input)`

Predict class labels for input samples.

**Parameters**

 - **model_input**: array-like or DataFrame of shape `(n_samples, n_features)`
     - Input data for prediction. Accepts NumPy arrays or Pandas DataFrame.

**Returns**

- **y_pred**: `np.ndarray` of shape `(n_samples,)`
    - Predicted class labels.

---

### `predict_proba(model_input)`

Predict class probabilities for input samples.

**Parameters**

 - **model_input**: array-like or DataFrame of shape `(n_samples, n_features)`
     - Input data for prediction. Accepts NumPy arrays or Pandas DataFrame.

**Returns**

- **y_proba**: `np.ndarray` of shape `(n_samples, n_classes)`
    - Predicted class probabilities.

---

### `score(X, y)`

Return the mean accuracy on the given test data and labels.

**Parameters**

 - **X**: array-like or DataFrame of shape `(n_samples, n_features)`
     - Input features for evaluation. Accepts NumPy arrays or Pandas DataFrame.
 - **y**: array-like or Series of shape `(n_samples,)`
     - True class labels. Accepts NumPy arrays or Pandas Series/DataFrame column.

**Returns**

- **score**: `float`
    - Accuracy score of the model predictions.

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

Instance of `GnosticBoostingClassifier` with loaded parameters.

---

## Example Usage

=== "Python"

    ```python
    from machinegnostics.models import GnosticBoostingClassifier

    # Initialize model
    model = GnosticBoostingClassifier(
        n_estimators=50,
        learning_rate=0.1,
        max_iter=5,
        verbose=True
    )

    # Fit the model
    model.fit(X, y)

    # Predict
    preds = model.predict(X[:5])
    print("Predictions:", preds)

    # Score
    acc = model.score(X, y)
    print(f'Accuracy: {acc:.4f}')
    ```

=== "Example Output"

    ![Gnostic Boosting Cls](image/gb_cls/1770044505488.png)

---

## Training History

If `history=True`, the model records detailed training history at each iteration, accessible via `model._history`.  This includes metrics like loss, residual entropy, and weight statistics, allowing users to visualize how the model adapts to the data quality over time.

---

## Notes

- **XGBoost Requirement**: This model requires `xgboost` to be installed in the environment.
- **Robustness**: The iterative gnostic weighting makes this model particularly robust against training data with label noise.

---

**Author:** Machine Gnostics Team  
**Date:** 2026
