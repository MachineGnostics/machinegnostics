"""
Machine Gnostics — Machine Gnostics Main Package
==================================================
Robust machine learning framework using gnostic weights (gw) that adaptively
down-weight outliers and noisy observations. No manual outlier removal required.

Submodules (maintainable, growable):
-------------------------------------
  • models      — regression, classification, clustering (LinearRegressor, etc.)
  • metrics     — statistical, regression, classification, gnostic (robr2, hc, gmmfe, ...)
  • magcal      — calibration engine: GnosticsWeights, ScaleParam, interval estimation
  • magnet      — magnet-phase calibration diagnostics (magcal phase 1-7 tests)
  • integration — MLflow tracking (optional dependency)

Key design rules for future growth:
  1. Each submodule has its own __init__ and imports cleanly.
  2. Metrics are grouped: stats, regression, classification, clustering, gnostic.
  3. All iterative models implement fit(), predict(), score() (scikit-learn style).
  4. Gnostic weights are computed inside magcal.GnosticsWeights.
  5. New metrics: add import block + add to __all__ + add health check in _check_imports.
  6. New submodule: add try/except import + availability flag + __all__ entry + get_available_components entry.

Version: 0.0.7
Author: Nirmal Parmar
Project: Machine Gnostics
"""

# =============================================================================
# Core Statistical Metrics
# =============================================================================
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
# Core Metrics — Covariance / Correlation
# =============================================================================
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
# Regression Metrics (robust + standard)
# =============================================================================
try:
    from .metrics.robr2 import robr2
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
try:
    from .metrics.silhouette_score import silhouette_score
    _CLUSTERING_METRICS_AVAILABLE = True
except ImportError as e:
    _CLUSTERING_METRICS_AVAILABLE = False
    _CLUSTERING_METRICS_IMPORT_ERROR = str(e)

# =============================================================================
# Gnostic-Specific Metrics (library-specific)
# =============================================================================
try:
    from .metrics.hc import hc
    from .metrics.gmmfe import gmmfe
    from .metrics.divi import divI
    from .metrics.evalmet import evalMet
    from .metrics.entropy import entropy
    _GNOSTIC_METRICS_AVAILABLE = True
except ImportError as e:
    _GNOSTIC_METRICS_AVAILABLE = False
    _GNOSTIC_METRICS_IMPORT_ERROR = str(e)

# =============================================================================
# Submodules (dot-access: mg.models, mg.metrics, ...)
# =============================================================================
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

try:
    from . import magnet
    _MAGNET_AVAILABLE = True
except ImportError as e:
    _MAGNET_AVAILABLE = False
    _MAGNET_IMPORT_ERROR = str(e)

try:
    from .integration import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError as e:
    _MLFLOW_AVAILABLE = False
    _MLFLOW_IMPORT_ERROR = str(e)

# =============================================================================
# Public API
# =============================================================================
__all__ = [
    'models', 'metrics', 'magcal', 'magnet', 'mlflow',
    'mean', 'median', 'std', 'variance',
    'auto_covariance', 'cross_covariance', 'correlation', 'auto_correlation',
    'robr2', 'r2_score', 'adjusted_r2_score',
    'mean_squared_error', 'mean_absolute_error', 'root_mean_squared_error',
    'accuracy_score', 'precision_score', 'recall_score', 'f1_score',
    'confusion_matrix', 'classification_report',
    'hc', 'gmmfe', 'divI', 'evalMet', 'entropy',
    'get_available_components',
]

# =============================================================================
# Module Metadata
# =============================================================================
__author__ = 'Nirmal Parmar'
__version__ = '0.0.7'

# =============================================================================
# Health Check / Component Availability
# =============================================================================

def get_available_components():
    """Return a dict of submodule/metric availability for diagnostics."""
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
        'magnet_submodule': _MAGNET_AVAILABLE,
        'mlflow_integration': _MLFLOW_AVAILABLE,
    }


def _check_imports():
    """Validate imports; warn on missing components (library keeps working)."""
    import warnings
    errors = []
    if not _STATS_AVAILABLE:
        errors.append(f"Statistical metrics: {_STATS_IMPORT_ERROR}")
    if not _COVAR_AVAILABLE:
        errors.append(f"Covariance/correlation: {_COVAR_IMPORT_ERROR}")
    if not _REGRESSION_METRICS_AVAILABLE:
        errors.append(f"Regression metrics: {_REGRESSION_METRICS_IMPORT_ERROR}")
    if not _CLASSIFICATION_METRICS_AVAILABLE:
        errors.append(f"Classification metrics: {_CLASSIFICATION_METRICS_IMPORT_ERROR}")
    if not _CLUSTERING_METRICS_AVAILABLE:
        errors.append(f"Clustering metrics: {_CLUSTERING_METRICS_IMPORT_ERROR}")
    if not _GNOSTIC_METRICS_AVAILABLE:
        errors.append(f"Gnostic metrics: {_GNOSTIC_METRICS_IMPORT_ERROR}")
    if not _MODELS_AVAILABLE:
        errors.append(f"Models: {_MODELS_IMPORT_ERROR}")
    if not _METRICS_MODULE_AVAILABLE:
        errors.append(f"Metrics: {_METRICS_MODULE_IMPORT_ERROR}")
    if not _MAGCAL_AVAILABLE:
        errors.append(f"Magcal: {_MAGCAL_IMPORT_ERROR}")
    if not _MAGNET_AVAILABLE:
        errors.append(f"Magnet: {_MAGNET_IMPORT_ERROR}")
    if errors:
        msg = ("Some Machine Gnostics components unavailable:\n  - " +
               "\n  - ".join(errors) + "\nLibrary functions with reduced capabilities.")
        warnings.warn(msg, ImportWarning, stacklevel=2)

_check_imports()
__all__.append('get_available_components')
