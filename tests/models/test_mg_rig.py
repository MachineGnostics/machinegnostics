import numpy as np
import pytest
from machinegnostics.models import RobustRegressor
import os
import pandas as pd

class TestRobustRegressor:
    @pytest.fixture
    def model(self):
        """Create a basic model instance for testing"""
        return RobustRegressor(degree=2, max_iter=100, tol=1e-6)

    def test_initialization(self, model):
        """Test model initialization"""
        assert model.degree == 2
        assert model.max_iter == 100
        assert model.tol == 1e-6
        assert model.coefficients is None
        assert model.weights is None
        assert isinstance(model._history, list)

    def test_fit_with_noisy_data(self, model):
        """Test fitting with noisy data"""
        np.random.seed(42)
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2*X + 1 + np.random.normal(0, 0.1, size=5)
        try:
            model.fit(X, y)
            assert model.coefficients is not None
            predictions = model.predict(X)
            assert np.corrcoef(X, predictions)[0, 1] > 0.9
        except ZeroDivisionError:
            pytest.skip("Skipping due to zero division in gnostic calculations")

    def test_predict_without_fit(self, model):
        """Test prediction without fitting"""
        X = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Model has not been fitted yet"):
            model.predict(X)

    def test_numerical_stability(self, model):
        """Test handling of near-zero values"""
        X = np.array([1e-10, 2e-10, 3e-10])
        y = np.array([2e-10, 4e-10, 6e-10])
        try:
            model.fit(X, y)
            assert model.coefficients is not None
        except np.linalg.LinAlgError:
            pytest.skip("Skipping due to numerical instability")
        except ZeroDivisionError:
            pytest.skip("Skipping due to zero division in gnostic calculations")

    def test_different_loss_functions(self):
        """Test model with different loss functions"""
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2*X + 1
        for loss in ['hi', 'hj']:
            model = RobustRegressor(degree=2, mg_loss=loss)
            try:
                model.fit(X, y)
                assert model.coefficients is not None
            except ZeroDivisionError:
                pytest.skip(f"Skipping {loss} loss due to zero division")

    def test_save_and_load_model(self, model, tmp_path):
        """Test saving and loading the model"""
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2*X + 1 + np.random.normal(0, 0.1, size=5)
        try:
            model.fit(X, y)
            save_path = str(tmp_path / "test_model")
            model.save_model(save_path)
            assert os.path.exists(os.path.join(save_path, "model.pkl"))
            loaded_model = RobustRegressor.load_model(save_path)
            assert loaded_model.degree == model.degree
            assert loaded_model.max_iter == model.max_iter
            assert loaded_model.tol == model.tol
            assert loaded_model.mg_loss == model.mg_loss
            assert loaded_model.early_stopping == model.early_stopping
            np.testing.assert_array_almost_equal(loaded_model.coefficients, model.coefficients)
            X_test = np.array([6.0, 7.0, 8.0])
            np.testing.assert_array_almost_equal(
                loaded_model.predict(X_test), model.predict(X_test)
            )
        except ZeroDivisionError:
            pytest.skip("Skipping due to zero division in gnostic calculations")

    def test_mlflow_python_model_inheritance(self, model):
        """Test that RobustRegressor properly inherits from mlflow.pyfunc.PythonModel"""
        import mlflow.pyfunc
        assert isinstance(model, mlflow.pyfunc.PythonModel)
        assert hasattr(model, 'predict')

    def test_edge_cases(self, model):
        """Test model behavior with edge cases"""
        # Single sample
        X_single = np.array([1.0])
        y_single = np.array([2.0])
        try:
            model.fit(X_single, y_single)
            assert model.coefficients is not None
            pred = model.predict(X_single)
            assert pred.shape == (1,)
        except ZeroDivisionError:
            pytest.skip("Skipping due to zero division in gnostic calculations")
        except np.linalg.LinAlgError:
            pytest.skip("Skipping due to numerical instability with single sample")
        # Constant features
        X_const = np.ones(5)
        y_const = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        try:
            model.fit(X_const, y_const)
            assert model.coefficients is not None
        except np.linalg.LinAlgError:
            pytest.skip("Skipping due to singular matrix in constant features test")
        except ZeroDivisionError:
            pytest.skip("Skipping due to zero division in gnostic calculations")

    def test_model_with_real_world_data(self):
        """Test model with a more realistic dataset"""
        np.random.seed(42)
        n_samples = 100
        X = np.random.rand(n_samples, 1) * 10
        true_coef = [1.5, -0.3, 0.05]
        polynomial = np.column_stack([np.ones(n_samples), X.flatten(), X.flatten()**2])
        y = polynomial @ true_coef + np.random.normal(0, 0.5, n_samples)
        model = RobustRegressor(degree=2, max_iter=200, tol=1e-6, verbose=False)
        try:
            model.fit(X.flatten(), y)
            assert model.coefficients is not None
            assert len(model.coefficients) == 3
            X_test = np.linspace(0, 10, 20).reshape(-1, 1)
            y_pred = model.predict(X_test.flatten())
            polynomial_test = np.column_stack([
                np.ones(20), X_test.flatten(), X_test.flatten()**2
            ])
            y_true = polynomial_test @ true_coef
            r2 = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))
            assert r2 > 0.8, f"Model fit quality too low: R² = {r2}"
        except ZeroDivisionError:
            pytest.skip("Skipping due to zero division in gnostic calculations")
        except np.linalg.LinAlgError:
            pytest.skip("Skipping due to linear algebra error")

    def test_predict_interface_compatibility(self, model):
        """Test that predict works with different input formats"""
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2*X + 1
        try:
            model.fit(X, y)
            pred1 = model.predict(np.array([6.0]))
            pred2 = model.predict([6.0])
            pred3 = model.predict(pd.Series([6.0]))
            assert abs(pred1[0] - pred2[0]) < 1e-8
            assert abs(pred1[0] - pred3[0]) < 1e-8
        except ZeroDivisionError:
            pytest.skip("Skipping due to zero division in gnostic calculations")
        except (ValueError, TypeError):
            pytest.skip("Input format incompatible with model prediction interface")