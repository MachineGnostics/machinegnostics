# KMeansClustering: Robust K-Means with Machine Gnostics

The `KMeansClustering` model performs clustering using robust, gnostic loss functions and adaptive sample weights. By integrating principles from Mathematical Gnostics, it offers superior resilience to outliers and non-Gaussian noise compared to standard K-Means implementations.

---

## Overview

Machine Gnostics `KMeansClustering` extends the traditional K-Means algorithm by introducing an iterative reweighting mechanism (`gnostic_weights`). This allows the model to dynamically down-weight contributions from anomalous data points during the centroid update phase, ensuring that cluster centers are not pulled away by outliers.

- **Robustness:** Minimizes the influence of outliers using gnostic loss functions.
- **Adaptive Weighting:** Automatically adjusts sample weights based on data quality.
- **Initialization Options:** Supports 'random' and 'kmeans++' initialization.
- **Optimization:** Includes convergence checks and early stopping.

---

## Key Features

- Robust clustering using gnostic loss functions ('hi' or 'hj')
- Adaptive sample weights for outlier suppression
- Detailed history tracking of optimization process
- Compatible with NumPy arrays and Pandas DataFrame/Series
- Easy model persistence (save/load)
- Standard Scikit-learn style API (fit, predict, score)

---

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `n_clusters` | `int` | `3` | The number of clusters to form. |
| `scale` | `str` \| `int` \| `float` | `'auto'` | Scaling method or value for gnostic calculations. |
| `max_iter` | `int` | `100` | Maximum number of optimization iterations. |
| `tolerance` | `float` | `1e-1` | Convergence tolerance. |
| `mg_loss` | `str` | `'hi'` | Gnostic loss function to use ('hi' or 'hj'). |
| `early_stopping` | `bool` | `True` | Whether to stop early upon convergence. |
| `verbose` | `bool` | `False` | Verbosity mode. |
| `data_form` | `str` | `'a'` | Data form: 'a' (additive) or 'm' (multiplicative). |
| `gnostic_characteristics` | `bool` | `False` | Compute and record gnostic characteristics. |
| `history` | `bool` | `True` | Whether to record optimization history. |
| `init` | `str` | `'random'` | Initialization method ('random' or 'kmeans++'). |

---

## Attributes

- **centroids**: `np.ndarray`
    - Fitted cluster centroids of shape `(n_clusters, n_features)`.
- **labels**: `np.ndarray`
    - Index of the cluster each sample belongs to.
- **weights**: `np.ndarray`
    - Final weights assigned to each sample after robust fitting.
- **params**: `list`
    - List of parameter snapshots (loss, weights, centroids) at each iteration.
- **_history**: `list`
    - Internal optimization history data.

---

## Methods

### `fit(X, y=None)`

Fit the robust k-means clustering model to the data.

This method iteratively optimizes cluster centroids and sample weights to minimize the influence of outliers.

**Parameters**

- **X**: array-like or DataFrame of shape `(n_samples, n_features)`
    - Input features. Accepts NumPy arrays or Pandas DataFrame.
- **y**: `Ignored`
    - Not used, present for API consistency.

**Returns**

- **self**: `KMeansClustering`
    - Returns the fitted model instance.

---

### `predict(model_input)`

Predict the closest cluster each sample in `model_input` belongs to.

**Parameters**

- **model_input**: array-like or DataFrame of shape `(n_samples, n_features)`
    - New data to predict. Accepts NumPy arrays or Pandas DataFrame.

**Returns**

- **labels**: `np.ndarray` of shape `(n_samples,)`
    - Index of the predicted cluster.

---

### `score(X, y=None)`

Compute the negative inertia score (sum of squared distances to closest centroid).

**Parameters**

- **X**: array-like or DataFrame of shape `(n_samples, n_features)`
    - Input features. Accepts NumPy arrays or Pandas DataFrame.
- **y**: `Ignored`
    - Not used.

**Returns**

- **score**: `float`
    - Negative inertia score. (Higher is better, consistent with sklearn).

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

Instance of `KMeansClustering` with loaded parameters.

---

## Example Usage

=== "Python"

    ```python
    import numpy as np
    from machinegnostics.models import KMeansClustering

    # Generate synthetic data with 3 clusters
    np.random.seed(42)
    X = np.concatenate([
        np.random.normal(0, 1, (50, 2)),
        np.random.normal(5, 1, (50, 2)),
        np.random.normal(10, 1, (50, 2))
    ])
    
    # Add some outliers
    X = np.vstack([X, np.random.uniform(-5, 15, (10, 2))])

    # Initialize and fit
    model = KMeansClustering(
        n_clusters=3,
        max_iter=50,
        verbose=True
    )
    model.fit(X)

    # Predict cluster labels
    labels = model.predict(X[:5])
    print("Labels:", labels)

    # Get centroids
    print("Centroids:", model.centroids)
    
    # Score
    score = model.score(X)
    print(f"Inertia Score: {score:.2f}")
    ```

=== "Example Output"

    ![KMeans Clustering](image/kmeans/1770046011433.png)


---

## Training History

If `history=True`, the model records detailed optimization history, including the evolution of centroids and weights. This is stored in `model.params` and `model._history`.

---

## Notes

- This model uses principles from Mathematical Gnostics to derive robust weights.
- It is especially useful in automated pipelines where data quality cannot be guaranteed.

---

**Author:** Nirmal Parmar  
