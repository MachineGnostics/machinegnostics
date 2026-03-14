# GnosticDecisionTreeClassifier: Robust Decision Tree with Machine Gnostics

The `GnosticDecisionTreeClassifier` implements a single Decision Tree Classifier enhanced with iterative gnostic reweighting. It is designed to handle outliers and varying data quality by identifying and down-weighting samples with high gnostic residual or uncertainty.

---

## Overview

Machine Gnostics `GnosticDecisionTreeClassifier` combines the interpretability of decision trees with the robustness of Mathematical Gnostics. It iteratively refines sample weights to improve classification performance in the presence of noise.

- **Robust Classification:** Identifies and down-weights samples with high gnostic residual/uncertainty.
- **Iterative Refinement:** Updates sample weights based on classification probabilities over iterations.
- **Data Quality Handling:** Automatically adjusts to data with outliers or non-Gaussian noise.
- **Flexible:** Supports configurable depth, splitting criteria, and gnostic iterations.
- **Event-Level Modeling:** Handles uncertainty at the level of individual data events.
- **Easy Model Persistence:** Save and load models with joblib.

---

## Key Features

- Robust classification using iterative gnostic reweighting
- Iterative refinement of sample weights
- Configurable tree depth and splitting criteria
- Identifies and handles outliers automatically
- Convergence-based early stopping
- Training history tracking for analysis
- Compatible with NumPy arrays and Pandas DataFrame/Series for input/output

---

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `max_depth` | `int` | `None` | Maximum depth of the tree. |
| `min_samples_split` | `int` | `2` | Minimum samples to split a node. |
| `gnostic_weights` | `bool` | `True` | Whether to use iterative gnostic weights. |
| `max_iter` | `int` | `10` | Maximum gnostic iterations. |
| `tolerance` | `float` | `1e-4` | Convergence tolerance. |
| `data_form` | `str` | `'a'` | Data form: 'a' (additive) or 'm' (multiplicative). |
| `verbose` | `bool` | `False` | Verbosity. |
| `random_state` | `int` | `None` | Random seed. |
| `history` | `bool` | `True` | Whether to record training history. |
| `scale` | `str` | `'auto'` | Scaling method for input features. |
| `early_stopping` | `bool` | `True` | Whether to stop training early if convergence is detected. |

---

## Attributes

- **gnostic_weights**: `bool`
    - Whether iterative gnostic weights were used.
- **max_depth**: `int`
    - Maximum depth of the tree.
- **max_iter**: `int`
    - Maximum number of iterations used.
- **_history**: `list`
    - List of dictionaries containing training history (if enabled).
- **tolerance, data_form, verbose, random_state, scale, early_stopping**
    - Configuration parameters as set at initialization.

---

## Methods

### `fit(X, y)`

Fit the model to the data.

This method trains the decision tree classifier using the provided input features and target labels. If `gnostic_weights` is True, it iteratively refines the model by reweighting samples based on gnostic residuals.

**Parameters**

 - **X**: array-like or DataFrame of shape `(n_samples, n_features)`
    - Input features. Accepts NumPy arrays or Pandas DataFrame.
- **y**: array-like or Series of shape `(n_samples,)`
    - Target labels. Accepts NumPy arrays or Pandas Series/DataFrame column.

**Returns**

- **self**: `GnosticDecisionTreeClassifier`
    - Returns the fitted model instance for chaining.

---

### `predict(model_input)`

Predict outcomes for new data.

**Parameters**

- **model_input**: array-like or DataFrame of shape `(n_samples, n_features)`
    - Input data for prediction. Accepts NumPy arrays or Pandas DataFrame.

**Returns**

- **y_pred**: `np.ndarray` of shape `(n_samples,)`
    - Predicted class labels.

---

### `score(X, y)`

Compute the accuracy score of the model on given data.

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

Instance of `GnosticDecisionTreeClassifier` with loaded parameters.

---

## Example Usage

=== "Python"

    ```python
    from machinegnostics.models import GnosticDecisionTreeClassifier

    # Initialize model
    model = GnosticDecisionTreeClassifier(
        max_depth=3,
        gnostic_weights=True,
        verbose=True
    )

    # Fit the model
    model.fit(X, y)

    # Predict
    preds = model.predict(X)

    # Score
    acc = model.score(X, y)
    print(f'Accuracy: {acc:.4f}')
    ```

=== "Example Output"

    ![Decision Tree Classifier](image/dt_cls/1770043380509.png)

---

## Training History

If `history=True`, the model records detailed training history at each iteration, accessible via `model._history`. This is particularly useful when `gnostic_weights=True` to observe how sample weights evolve over iterations.

---

## Notes

- This model is resilient to outliers due to the iterative gnostic reweighting mechanism.
- Suitable for datasets where data quality varies or label noise is present.

---

**Author:** Nirmal Parmar  
