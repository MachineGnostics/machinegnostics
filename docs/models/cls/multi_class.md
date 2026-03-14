# MulticlassClassifier: Robust Multiclass Classification with Machine Gnostics

The `MulticlassClassifier` is a robust multiclass classification model built on the Machine Gnostics framework. It provides feature-rich classification with polynomial feature expansion, softmax activation, and detailed history tracking, designed to be resilient to outliers and improve model stability.

---

## Overview

Machine Gnostics `MulticlassClassifier` brings deterministic, event-level modeling to multiclass problems. By leveraging gnostic algebra and geometry, it provides robust, interpretable, and reproducible results.

- **Deterministic & Finite:** No randomness or probability; all computations are reproducible.
- **Event-Level Modeling:** Handles uncertainty and error at the level of individual data events.
- **Robust:** Designed to be robust against outliers, corrupted data, and distributional shifts.
- **Flexible:** Supports polynomial feature expansion and automatic class detection.
- **mlflow Integration:** For experiment tracking and deployment.
- **Easy Model Persistence:** Save and load models with joblib.

---

## Key Features

- Multiclass classification using softmax activation
- Polynomial feature expansion up to a user-specified degree
- Gnostic weights for robust handling of outliers
- Automatic detection of number of classes
- Iterative optimization with early stopping and convergence tolerance
- Training history tracking for analysis and visualization
- Compatible with NumPy arrays and Pandas DataFrame/Series for input/output

---

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `degree` | `int` | `1` | Degree of polynomial features to use for input expansion. |
| `max_iter` | `int` | `100` | Maximum number of iterations for the optimization algorithm. |
| `tolerance` | `float` | `1e-1` | Tolerance for convergence. |
| `early_stopping` | `bool` | `True` | Whether to stop training early if convergence is detected. |
| `verbose` | `bool` | `False` | If True, prints detailed logs during training. |
| `scale` | `str` \| `int` \| `float` | `'auto'` | Scaling method for gnostic weight calculations. |
| `data_form` | `str` | `'a'` | Internal data representation format. |
| `gnostic_characteristics` | `bool` | `False` | If True, computes and records gnostic characteristics. |
| `history` | `bool` | `True` | If True, records the optimization history for analysis. |

---

## Attributes

- **coefficients**: `np.ndarray`
    - Fitted model coefficients after training.
- **weights**: `np.ndarray`
    - Sample weights used during training.
- **num_classes**: `int`
    - Number of unique classes in the training data.
- **cross_entropy_loss**: `float`
    - Cross-entropy loss computed during training.
- **params**: `list of dict`
    - List of model parameters (for compatibility and inspection).
- **_history**: `list`
    - List of dictionaries containing training history (loss, coefficients, entropy, etc.).
- **degree, max_iter, tolerance, early_stopping, verbose, scale, data_form, gnostic_characteristics**
    - Configuration parameters as set at initialization.

---

## Methods

### `fit(X, y)`

Fit the multiclass classifier to the data.

This method trains the classifier using the provided input features and target labels. It supports polynomial feature expansion, softmax activation, and early stopping.

**Parameters**

- **X**: `array-like` or `DataFrame` of shape `(n_samples, n_features)`
    - Input features for training. Accepts NumPy arrays or Pandas DataFrame.
- **y**: `array-like` of shape `(n_samples,)`
    - Target labels for training (class integers starting from 0). Accepts NumPy arrays or Pandas Series/DataFrame column.

**Returns**

- **self**: `MulticlassClassifier`
    - Returns the fitted model instance for chaining.

---

### `predict(model_input)`

Predict class labels for new data.

**Parameters**

- **model_input**: `array-like` or `DataFrame` of shape `(n_samples, n_features)`
    - Input data for prediction. Accepts NumPy arrays or Pandas DataFrame.

**Returns**

- **y_pred**: `np.ndarray` of shape `(n_samples,)`
    - Predicted class labels (integers).

---

### `predict_proba(model_input)`

Predict class probabilities for new data.

**Parameters**

- **model_input**: `array-like` or `DataFrame` of shape `(n_samples, n_features)`
    - Input data for probability prediction. Accepts NumPy arrays or Pandas DataFrame.

**Returns**

- **y_proba**: `np.ndarray` of shape `(n_samples, n_classes)`
    - Predicted probabilities for each class.

---

### `score(X, y)`

Compute the accuracy score of the model on given data.

**Parameters**

- **X**: `array-like` or `DataFrame` of shape `(n_samples, n_features)`
    - Input features for evaluation. Accepts NumPy arrays or Pandas DataFrame.
- **y**: `array-like` of shape `(n_samples,)`
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

Instance of `MulticlassClassifier` with loaded parameters.

---

## Example Usage

=== "Python"

    ```python
    import numpy as np
    from machinegnostics.models import MulticlassClassifier

    # Generate synthetic multiclass data (3 classes)
    np.random.seed(42)
    n_samples = 150
    X = np.random.randn(n_samples, 2)
    # Assign classes based on regions
    y = np.zeros(n_samples, dtype=int)
    y[X[:, 0] > 0.5] = 1
    y[X[:, 1] < -0.5] = 2

    # Initialize model
    model = MulticlassClassifier(
        degree=2,
        max_iter=100,
        verbose=True,
        tolerance=1e-3
    )

    # Fit the model
    model.fit(X, y)

    # Predict
    X_test = np.array([[1.0, 1.0], [-1.0, -1.0], [0.0, -1.0]])
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Score
    acc = model.score(X, y)
    print(f'Accuracy: {acc:.4f}')
    ```

=== "Example Output"

    ![Multi Class Classification](image/multi_class/1770042931669.png)

---

## Training History

If `history=True`, the model records detailed training history at each iteration, accessible via `model.params` and `model._history`. Each entry contains details like cross-entropy loss, coefficients, and entropy.

---

## Notes

- The model automatically detects the number of classes from the training data.
- Uses softmax activation for multiclass probability estimation.

---

**Author:** Nirmal Parmar  
