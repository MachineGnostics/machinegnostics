import numpy as np
import pandas as pd
import pytest
from src.metrics import robr2, divI, evalMet, gmmfe
from src.magcal import gautocovariance, gcorrelation, gcovariance, gmedian, gmodulus

class TestMetricsFunctions:
    
    @pytest.fixture
    def sample_data(self):
        """Fixture providing sample data for metrics testing"""
        np.random.seed(42)
        y_true = np.random.normal(0, 1, 100)
        y_pred = y_true + np.random.normal(0, 0.5, 100)
        return y_true, y_pred
    
    @pytest.fixture
    def outlier_data(self):
        """Fixture providing data with outliers"""
        np.random.seed(42)
        y_true = np.random.normal(0, 1, 100)
        y_pred = y_true + np.random.normal(0, 0.5, 100)
        # Add outliers
        y_true[0] = 10.0
        y_pred[0] = 0.0  # Large prediction error
        return y_true, y_pred
    
    def test_robr2(self, sample_data, outlier_data):
        """Test robust R² calculation"""
        y_true, y_pred = sample_data
        y_true_out, y_pred_out = outlier_data
        
        # Test basic functionality
        r2 = robr2(y_true, y_pred)
        assert isinstance(r2, float)
        assert r2 <= 1.0, "R² should be <= 1"
        
        # Test outlier robustness
        std_r2_clean = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))
        std_r2_outlier = 1 - (np.sum((y_true_out - y_pred_out)**2) / np.sum((y_true_out - np.mean(y_true_out))**2))
        
        rob_r2_clean = robr2(y_true, y_pred)
        rob_r2_outlier = robr2(y_true_out, y_pred_out)
        
        # Robust R² should be less affected by outliers
        std_diff = abs(std_r2_clean - std_r2_outlier)
        rob_diff = abs(rob_r2_clean - rob_r2_outlier)
        
        assert rob_diff < std_diff, "Robust R² should be less sensitive to outliers"
        
        # Test with perfect predictions
        perfect_y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.isclose(robr2(perfect_y, perfect_y), 1.0), "R² should be 1 for perfect predictions"
    