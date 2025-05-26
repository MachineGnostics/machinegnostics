import numpy as np
import pytest
from machinegnostics.models import LogisticRegressor

def test_logistic_regressor_basic():
    np.random.seed(42)
    X = np.random.randn(200, 3)
    true_coef = np.array([1.5, -2.0, 0.5])
    logits = X @ true_coef
    probs = 1 / (1 + np.exp(-logits))
    y = (probs > 0.5).astype(int)

    model = LogisticRegressor(degree=1, verbose=False)
    model.fit(X, y)

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    assert y_pred.shape == y.shape, "Predicted labels shape mismatch"
    assert y_proba.shape == y.shape, "Predicted probabilities shape mismatch"
    assert np.all((y_proba >= 0) & (y_proba <= 1)), "Probabilities out of bounds"
    accuracy = np.mean(y_pred == y)
    assert accuracy > 0.8, "Model accuracy too low on synthetic data"

def test_logistic_regressor_predict_unseen():
    np.random.seed(0)
    X_train = np.random.randn(100, 2)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
    X_test = np.random.randn(20, 2)

    model = LogisticRegressor(degree=1, verbose=False)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    assert y_pred.shape == (20,)
    assert y_proba.shape == (20,)
    assert np.all((y_proba >= 0) & (y_proba <= 1))

def test_logistic_regressor_invalid_input_shape():
    np.random.seed(2)
    X = np.random.randn(30, 3)
    y = np.random.randint(0, 2, size=30)

    model = LogisticRegressor(degree=1, verbose=False)
    model.fit(X, y)
    X_bad = np.random.randn(10, 2)  # Wrong feature size
    with pytest.raises(Exception):
        model.predict(X_bad)

def test_logistic_regressor_probability_sum():
    np.random.seed(3)
    X = np.random.randn(60, 2)
    y = np.random.randint(0, 2, size=60)

    model = LogisticRegressor(degree=1, verbose=False)
    model.fit(X, y)
    proba = model.predict_proba(X)
    # For binary classification, proba should be between 0 and 1
    assert np.all((proba >= 0) & (proba <= 1))

# To run with pytest, save as test_logreg.py and run: pytest test_logreg.py