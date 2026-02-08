"""
Machine Gnostics: Robust Machine Learning with Gnostic Weights
================================================================

Machine Gnostics is a comprehensive framework for robust machine learning that
automatically handles outliers and noisy data through gnostic weight mechanisms.

Key Innovation:
---------------
The framework introduces **gnostic weights (gw)** that adaptively down-weight
outliers and poorly-fitting data points, leading to more robust models without
manual outlier detection or data preprocessing.

Main Components:
----------------
1. **Models**: Regression, classification, and clustering with gnostic weights
   - LinearRegressor, PolynomialRegressor
   - LogisticRegressor, MulticlassClassifier
   - KMeansClustering

2. **Metrics**: Custom robust metrics and standard ML evaluation metrics
   - Robust R² (robr2), standard metrics (MSE, MAE, RMSE, R²)
   - Classification metrics (accuracy, precision, recall, F1)
   - Gnostic-specific metrics (hc, gmmfe, divI, evalMet)

3. **Calibration (magcal)**: Gnostic weight computation and calibration
4. **Integration**: MLflow integration for experiment tracking

Quick Start:
------------
    import machinegnostics as mg
    
    # Direct metric access
    r2 = mg.r2_score(y_true, y_pred)
    acc = mg.accuracy_score(y_true, y_pred)
    
    # Model usage
    from machinegnostics.models import LinearRegressor
    model = LinearRegressor()
    model.fit(X, y)
    
    # Clustering
    from machinegnostics.models import KMeansClustering
    kmeans = mg.models.KMeansClustering(n_clusters=3)
    kmeans.fit(X)

Usage Patterns:
---------------
    # Pattern 1: Direct function access
    import machinegnostics as mg
    result = mg.mean(data)
    score = mg.robr2(y_true, y_pred)
    
    # Pattern 2: Submodule access
    from machinegnostics import models, metrics
    model = models.LinearRegressor()
    r2 = metrics.r2_score(y_true, y_pred)
    
    # Pattern 3: Specific imports
    from machinegnostics import robr2, KMeansClustering

Architecture:
-------------
- **models**: Machine learning models with gnostic weights
- **metrics**: Evaluation metrics (robust and standard)
- **magcal**: Calibration and gnostic weight computation
- **integration**: Third-party integrations (MLflow, etc.)

Notes:
------
- All models implement fit(), predict(), and score() methods
- Gnostic weights are computed automatically during training
- History tracking available for all iterative models
- Compatible with scikit-learn API conventions

References:
-----------
For more information on gnostic weights and the mathematical framework,
see the documentation and published papers.

Version: 1.0.0
Author: Machine Gnostics Team
"""

# =============================================================================
# Core Metrics - Statistical Functions
# =============================================================================
# Basic statistical measures for data analysis

try:
    from .metrics.mean import mean
    from .metrics.median import median
    from .metrics.std import std
    from .metrics.variance import variance
    _STATS_AVAILABLE = True
except ImportError as e:
    _STATS_AVAILABLE = False
    _STATS_IMPORT_ERROR = str(e)

# =============================================================================
# Core Metrics - Covariance and Correlation
# =============================================================================
# Measures of relationship between variables

try:
    from .metrics.auto_covariance import auto_covariance
    from .metrics.cross_variance import cross_covariance
    from .metrics.correlation import correlation
    from .metrics.auto_correlation import auto_correlation
    _COVAR_AVAILABLE = True
except ImportError as e:
    _COVAR_AVAILABLE = False
    _COVAR_IMPORT_ERROR = str(e)

# =============================================================================
# Regression Metrics
# =============================================================================
# Standard and robust metrics for regression evaluation

try:
    from .metrics.robr2 import robr2  # Robust R² (gnostic)
    from .metrics.r2 import r2_score, adjusted_r2_score
    from .metrics.mse import mean_squared_error
    from .metrics.mae import mean_absolute_error
    from .metrics.rmse import root_mean_squared_error
    _REGRESSION_METRICS_AVAILABLE = True
except ImportError as e:
    _REGRESSION_METRICS_AVAILABLE = False
    _REGRESSION_METRICS_IMPORT_ERROR = str(e)

# =============================================================================
# Classification Metrics
# =============================================================================
# Standard metrics for classification evaluation

try:
    from .metrics.accuracy import accuracy_score
    from .metrics.precision import precision_score
    from .metrics.recall import recall_score
    from .metrics.f1_score import f1_score
    from .metrics.conf_matrix import confusion_matrix
    from .metrics.cls_report import classification_report
    _CLASSIFICATION_METRICS_AVAILABLE = True
except ImportError as e:
    _CLASSIFICATION_METRICS_AVAILABLE = False
    _CLASSIFICATION_METRICS_IMPORT_ERROR = str(e)


# =============================================================================
# Clustering Metrics
# =============================================================================
# Standard metrics for clustering evaluation

try:
    from .metrics.silhouette_score import silhouette_score
    _CLUSTERING_METRICS_AVAILABLE = True
except ImportError as e:
    _CLUSTERING_METRICS_AVAILABLE = False
    _CLUSTERING_METRICS_IMPORT_ERROR = str(e)

# =============================================================================
# Gnostic-Specific Metrics
# =============================================================================
# Custom metrics for gnostic model evaluation

try:
    from .metrics.hc import hc  # Harmonic characteristics
    from .metrics.gmmfe import gmmfe  # Gnostic model metric
    from .metrics.divi import divI  # Diversity index
    from .metrics.evalmet import evalMet  # Evaluation metric
    from .metrics.entropy import entropy # Gnostic Entropy
    _GNOSTIC_METRICS_AVAILABLE = True
except ImportError as e:
    _GNOSTIC_METRICS_AVAILABLE = False
    _GNOSTIC_METRICS_IMPORT_ERROR = str(e)

# =============================================================================
# Submodule Imports
# =============================================================================
# Import submodules for dot notation access (e.g., mg.models.LinearRegressor)

try:
    from . import models
    _MODELS_AVAILABLE = True
except ImportError as e:
    _MODELS_AVAILABLE = False
    _MODELS_IMPORT_ERROR = str(e)

try:
    from . import metrics
    _METRICS_MODULE_AVAILABLE = True
except ImportError as e:
    _METRICS_MODULE_AVAILABLE = False
    _METRICS_MODULE_IMPORT_ERROR = str(e)

try:
    from . import magcal
    _MAGCAL_AVAILABLE = True
except ImportError as e:
    _MAGCAL_AVAILABLE = False
    _MAGCAL_IMPORT_ERROR = str(e)

# =============================================================================
# Integration Support
# =============================================================================
# Third-party integrations (MLflow, etc.)

try:
    from .integration import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError as e:
    _MLFLOW_AVAILABLE = False
    _MLFLOW_IMPORT_ERROR = str(e)


# =============================================================================
# Public API Definition
# =============================================================================

__all__ = [
    # Submodules
    'models',
    'metrics',
    'magcal',
    'mlflow',
    
    # Statistical metrics
    'mean',
    'median',
    'std',
    'variance',
    
    # Covariance/Correlation
    'auto_covariance',
    'cross_covariance',
    'correlation',
    'auto_correlation',
    
    # Regression metrics
    'robr2',
    'r2_score',
    'adjusted_r2_score',
    'mean_squared_error',
    'mean_absolute_error',
    'root_mean_squared_error',
    
    # Classification metrics
    'accuracy_score',
    'precision_score',
    'recall_score',
    'f1_score',
    'confusion_matrix',
    'classification_report',
    
    # Gnostic metrics
    'hc',
    'gmmfe',
    'divI',
    'evalMet',
    'entropy'
    

]

# =============================================================================
# Module Metadata
# =============================================================================

__author__ = 'Nirmal Parmar'

# check version with tomal file
__version__ = '0.0.3'

# =============================================================================
# Import Validation and Health Check
# =============================================================================

def _check_imports():
    """
    Validate all imports and provide helpful diagnostics.
    
    This function runs automatically on module import and warns about
    any missing components while still allowing the module to function
    with available components.
    """
    import warnings
    errors = []
    
    # Check core functionality
    if not _STATS_AVAILABLE:
        errors.append(f"Statistical metrics unavailable: {_STATS_IMPORT_ERROR}")
    
    if not _COVAR_AVAILABLE:
        errors.append(f"Covariance/correlation metrics unavailable: {_COVAR_IMPORT_ERROR}")
    
    if not _REGRESSION_METRICS_AVAILABLE:
        errors.append(f"Regression metrics unavailable: {_REGRESSION_METRICS_IMPORT_ERROR}")
    
    if not _CLASSIFICATION_METRICS_AVAILABLE:
        errors.append(f"Classification metrics unavailable: {_CLASSIFICATION_METRICS_IMPORT_ERROR}")

    if not _CLUSTERING_METRICS_AVAILABLE:
        errors.append(f"Clustering metrics unavailable: {_CLUSTERING_METRICS_IMPORT_ERROR}")
    
    if not _GNOSTIC_METRICS_AVAILABLE:
        errors.append(f"Gnostic-specific metrics unavailable: {_GNOSTIC_METRICS_IMPORT_ERROR}")
    
    # Check submodules
    if not _MODELS_AVAILABLE:
        errors.append(f"Models submodule unavailable: {_MODELS_IMPORT_ERROR}")
    
    if not _METRICS_MODULE_AVAILABLE:
        errors.append(f"Metrics submodule unavailable: {_METRICS_MODULE_IMPORT_ERROR}")
    
    if not _MAGCAL_AVAILABLE:
        errors.append(f"Magcal (calibration) submodule unavailable: {_MAGCAL_IMPORT_ERROR}")
    
    # Check integrations (optional)
    if not _MLFLOW_AVAILABLE:
        # MLflow is optional, so just note it
        pass
    
    # Emit warnings if there are errors
    if errors:
        warning_msg = (
            "Some Machine Gnostics components could not be imported:\n" +
            "\n".join(f"  - {err}" for err in errors) +
            "\n\nThe module will function with reduced capabilities."
        )
        warnings.warn(warning_msg, ImportWarning, stacklevel=2)

def get_available_components():
    """
    Return a dictionary showing which components are available.
    
    Returns:
    --------
    dict
        Dictionary with component names as keys and availability status as values.
    
    Example:
    --------
        >>> import machinegnostics as mg
        >>> mg.get_available_components()
        {'stats': True, 'models': True, 'metrics': True, ...}
    """
    return {
        'statistical_metrics': _STATS_AVAILABLE,
        'covariance_correlation': _COVAR_AVAILABLE,
        'regression_metrics': _REGRESSION_METRICS_AVAILABLE,
        'classification_metrics': _CLASSIFICATION_METRICS_AVAILABLE,
        'clustering_metrics': _CLUSTERING_METRICS_AVAILABLE,
        'gnostic_metrics': _GNOSTIC_METRICS_AVAILABLE,
        'models_submodule': _MODELS_AVAILABLE,
        'metrics_submodule': _METRICS_MODULE_AVAILABLE,
        'magcal_submodule': _MAGCAL_AVAILABLE,
        'mlflow_integration': _MLFLOW_AVAILABLE,
    }

# Run import validation
_check_imports()

# Add helper to __all__
__all__.append('get_available_components')

# Clean up namespace
del _check_imports