"""
Machine Gnostics Models Module
================================

This module provides access to all machine learning models in the Machine Gnostics
framework. All models implement gnostic weight mechanisms for improved robustness
and interpretability.

Model Categories:
-----------------
1. **Regression Models**: Linear and polynomial regression with gnostic weights
2. **Classification Models**: Logistic regression and multiclass classification
3. **Clustering Models**: K-means clustering with adaptive gnostic weighting
4. **Support Utilities**: Cross-validation and data splitting tools

Key Features:
-------------
- Gnostic weight multiplication (weights * gw) for robust learning
- Automatic outlier handling through adaptive weighting
- History tracking for all model iterations
- Consistent API across all model types

Usage Example:
--------------
    from machinegnostics.models import LinearRegressor, KMeansClustering
    
    # Regression
    model = LinearRegressor()
    model.fit(X, y)
    predictions = model.predict(X_test)
    
    # Clustering
    kmeans = KMeansClustering(n_clusters=3)
    kmeans.fit(X)
    labels = kmeans.predict(X_test)

Notes:
------
- All models follow standard API conventions
- Models support fit(), predict(), and score() methods
- History attributes provide detailed iteration information
- Gnostic weights are automatically computed and applied

"""

# =============================================================================
# Regression Models
# =============================================================================
# Linear and polynomial regression with gnostic weight mechanisms

try:
    from machinegnostics.models.regression.linear_regressor import LinearRegressor
    from machinegnostics.models.regression.polynomial_regressor import PolynomialRegressor
    from machinegnostics.models.cart.tree_regressor import GnosticDecisionTreeRegressor
    from machinegnostics.models.cart.random_forest_regressor import GnosticRandomForestRegressor
    from machinegnostics.models.cart.boosting_regressor import GnosticBoostingRegressor
    # forecasting models
    from machinegnostics.models.regression.auto_regressor import AutoRegressor
    from machinegnostics.models.regression.arima import ARIMA
    from machinegnostics.models.regression.sarima import SARIMA
    _REGRESSION_AVAILABLE = True
except ImportError as e:
    _REGRESSION_AVAILABLE = False
    _REGRESSION_IMPORT_ERROR = str(e)

# =============================================================================
# Classification Models
# =============================================================================
# Binary and multiclass classification with gnostic weights

try:
    from machinegnostics.models.classification.logistic_regressor import LogisticRegressor
    from machinegnostics.models.classification.gnostic_multiclass_classifier import MulticlassClassifier
    from machinegnostics.models.cart.tree_classifier import GnosticDecisionTreeClassifier
    from machinegnostics.models.cart.random_forest_classifier import GnosticRandomForestClassifier
    from machinegnostics.models.cart.boosting_classifier import GnosticBoostingClassifier
    _CLASSIFICATION_AVAILABLE = True
except ImportError as e:
    _CLASSIFICATION_AVAILABLE = False
    _CLASSIFICATION_IMPORT_ERROR = str(e)

# =============================================================================
# Clustering Models
# =============================================================================
# Unsupervised learning with adaptive gnostic weighting

try:
    from machinegnostics.models.clustering.kmeans_clustering import KMeansClustering
    from machinegnostics.models.clustering.local_clustering import GnosticLocalClustering
    _CLUSTERING_AVAILABLE = True
except ImportError as e:
    _CLUSTERING_AVAILABLE = False
    _CLUSTERING_IMPORT_ERROR = str(e)

# =============================================================================
# Support Utilities
# =============================================================================
# Cross-validation and data preprocessing tools

try:
    from machinegnostics.models.cross_validation import CrossValidator
    from machinegnostics.models.data_split import train_test_split
    _SUPPORT_AVAILABLE = True
except ImportError as e:
    _SUPPORT_AVAILABLE = False
    _SUPPORT_IMPORT_ERROR = str(e)

# =============================================================================
# Public API Definition
# =============================================================================
# Explicitly define what gets exported with "from machinegnostics.models import *"

__all__ = [
    # Regression
    'LinearRegressor',
    'PolynomialRegressor',
    'GnosticDecisionTreeRegressor',
    'GnosticRandomForestRegressor',
    'GnosticBoostingRegressor',
    'AutoRegressor',
    'ARIMA',
    'SARIMA',
    # Classification
    'LogisticRegressor',
    'MulticlassClassifier',
    'GnosticDecisionTreeClassifier',
    'GnosticRandomForestClassifier',
    'GnosticBoostingClassifier',
    # Clustering
    'KMeansClustering',
    'GnosticLocalClustering',
    # Support
    'CrossValidator',
    'train_test_split',
]

# =============================================================================
# Module Metadata
# =============================================================================

__author__ = 'Nirmal Parmar'

# =============================================================================
# Import Validation
# =============================================================================
# Provide helpful error messages if imports fail

def _check_imports():
    """
    Check if all model imports were successful and provide helpful error messages.
    
    This function is called automatically when the module is imported.
    """
    errors = []
    
    if not _REGRESSION_AVAILABLE:
        errors.append(f"Regression models unavailable: {_REGRESSION_IMPORT_ERROR}")
    
    if not _CLASSIFICATION_AVAILABLE:
        errors.append(f"Classification models unavailable: {_CLASSIFICATION_IMPORT_ERROR}")
    
    if not _CLUSTERING_AVAILABLE:
        errors.append(f"Clustering models unavailable: {_CLUSTERING_IMPORT_ERROR}")
    
    if not _SUPPORT_AVAILABLE:
        errors.append(f"Support utilities unavailable: {_SUPPORT_IMPORT_ERROR}")
    
    if errors:
        import warnings
        warning_msg = "Some models could not be imported:\n" + "\n".join(f"  - {err}" for err in errors)
        warnings.warn(warning_msg, ImportWarning, stacklevel=2)

# Run import validation
_check_imports()

# Clean up namespace (remove internal variables from public API)
del _check_imports