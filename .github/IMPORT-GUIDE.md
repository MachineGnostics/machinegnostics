# Machine Gnostics - Import Guide & Standards

## Design Principle

Machine Gnostics follows a **shallow import hierarchy** to provide users with intuitive, easy-to-use imports while maintaining clean module organization.

**Maximum Import Depth:** `machinegnostics.[magnet|magcal|model|metrics|data|integrations]`

This allows users to import exactly what they need without diving deep into internal module structures.

---

## Recommended Import Patterns

### Pattern 1: Module-Level Imports (Recommended)
```python
import machinegnostics as mg

# MAGNET framework
from mg.magnet import Dense, Fi, Loss, Sequential
from mg.magnet.activations import sech_fidelity, tanh_irrelevance
from mg.magnet.losses import InfidelityLoss, FidelityLoss

# Magcal calibration
from mg.magcal import Characteristics, CalibrationModel
from mg.magcal.data_conversion import convert_sensor_data

# Models
from mg.model import RandomForest, SVM, KMeans

# Metrics
from mg.metrics import accuracy_score, precision_score, fidelity_score

# Data utilities
from mg.data import load_diagnostic_data, preprocess

# Integrations
from mg.integrations import MLflowTracker, CloudAdapter
```

### Pattern 2: Direct Function Access
```python
import machinegnostics as mg

# Access top-level functions directly
score = mg.metrics.mean_squared_error(y_true, y_pred)
model = mg.model.train_classifier(X_train, y_train)
```

### Pattern 3: Submodule Imports (when needed)
```python
# Only when accessing specific sub-components
from mg.magnet.core import gnostic_backprop
from mg.magcal.gdf import calculate_distribution_function
```

---

## Directory Structure & Import Paths

```
machinegnostics/
│
├── __init__.py                    # Top-level exports
│
├── magnet/                        # MAGNET framework
│   ├── __init__.py               # Exports: Dense, Fi, Loss, Sequential, etc.
│   ├── layers/
│   ├── activations/
│   ├── losses/
│   ├── core/
│   └── basic/
│
├── magcal/                        # Calibration module
│   ├── __init__.py               # Exports: Characteristics, CalibrationModel, etc.
│   ├── characteristics.py
│   ├── data_conversion.py
│   ├── gdf/
│   └── util/
│
├── model/                         # ML Models
│   ├── __init__.py               # Exports: RandomForest, SVM, KMeans, etc.
│   ├── classifiers/
│   ├── regressors/
│   └── ensemble/
│
├── metrics/                       # Evaluation metrics
│   ├── __init__.py               # Exports: accuracy_score, fidelity_score, etc.
│   ├── diagnostic_metrics.py
│   ├── performance_metrics.py
│   └── validators.py
│
├── data/                          # Data utilities
│   ├── __init__.py               # Exports: load_data, preprocess, etc.
│   ├── loaders.py
│   └── preprocessing.py
│
└── integrations/                  # Cloud/MLflow integrations
    ├── __init__.py               # Exports: MLflowTracker, CloudAdapter, etc.
    ├── mlflow_config.py
    ├── cloud_adapters/
    └── deployment/
```

---

## __init__.py Guidelines for Each Module

### Level 0: Top-Level `machinegnostics/__init__.py`

```python
"""Machine Gnostics framework for robust machine learning."""

# Import major submodules to enable mg.magnet, mg.magcal, etc.
from . import magnet
from . import magcal
from . import model
from . import metrics
from . import data
from . import integrations

# Optionally expose frequently-used items at top level
from .metrics import accuracy_score, precision_score, mean_squared_error
from .data import load_diagnostic_data

__all__ = [
    "magnet",
    "magcal",
    "model",
    "metrics",
    "data",
    "integrations",
    # Top-level convenience exports
    "accuracy_score",
    "precision_score", 
    "mean_squared_error",
    "load_diagnostic_data",
]
```

### Level 1: Subsystem `__init__.py` (e.g., `magnet/__init__.py`)

```python
"""MAGNET: Machine Gnostic Neural Network Environment/Toolkit"""

# Import and expose key classes/functions from submodules
from .layers import Dense, Flatten, BatchNorm
from .activations import Fi, Hi, sech_fidelity, tanh_irrelevance
from .losses import InfidelityLoss, FidelityLoss
from .core import Sequential, Model, GnosticRuntime
from .basic import validate_concept  # For testing

__all__ = [
    # Layers
    "Dense",
    "Flatten",
    "BatchNorm",
    # Activations (Fi and Hi are gnostic activation functions)
    "Fi",
    "Hi",
    "sech_fidelity",
    "tanh_irrelevance",
    # Losses
    "InfidelityLoss",
    "FidelityLoss",
    # Core
    "Sequential",
    "Model",
    "GnosticRuntime",
    # Utilities
    "validate_concept",
]
```

### Level 2: Sub-subsystem `__init__.py` (e.g., `magnet/layers/__init__.py`)

```python
"""MAGNET layers: Standard and gnostic neural network layers"""

from .base import BaseLayer
from .dense import Dense, gDense
from .idense import Fi, iDense
from .irrelevance import Hi
from .batchnorm import BatchNorm
from .flatten import Flatten

__all__ = [
    "BaseLayer",
    "Dense",
    "gDense",
    "Fi",
    "iDense",
    "Hi",
    "BatchNorm",
    "Flatten",
]
```

---

## Import Usage Examples

### Example 1: Building a MAGNET Sequential Model

```python
import machinegnostics as mg
from mg.magnet import Dense, Fi, Sequential
from mg.magnet.losses import InfidelityLoss
from mg.metrics import fidelity_score

# Build model
model = mg.magnet.Sequential([
    mg.magnet.Dense(64, input_dim=28*28),
    mg.magnet.Fi(32),
    mg.magnet.Fi(10),
])

# Compile with gnostic loss
model.compile(
    loss=mg.magnet.InfidelityLoss(),
    optimizer='adam'
)

# Train
model.fit(X_train, y_train, epochs=10)

# Evaluate
predictions = model.predict(X_test)
score = mg.metrics.fidelity_score(y_test, predictions)
```

### Example 2: Magcal Calibration Workflow

```python
import machinegnostics as mg
from mg.magcal import Characteristics, CalibrationModel
from mg.data import load_diagnostic_data

# Load data
X, y = mg.data.load_diagnostic_data('bearing_fault')

# Create characteristics
chars = mg.magcal.Characteristics(
    sensor_type='magnetic',
    frequency_range=(0, 5000)
)

# Calibrate
calibrator = mg.magcal.CalibrationModel(chars)
calibration_params = calibrator.calibrate(X)

# Apply calibration
X_calibrated = mg.magcal.data_conversion.apply_calibration(X, calibration_params)
```

### Example 3: ML Model Training

```python
import machinegnostics as mg

# Load data
X, y = mg.data.load_diagnostic_data('condition_monitoring')

# Train classifier
clf = mg.model.RandomForest(n_estimators=100)
clf.fit(X, y)

# Evaluate
predictions = clf.predict(X_test)
accuracy = mg.metrics.accuracy_score(y_test, predictions)
precision = mg.metrics.precision_score(y_test, predictions)

print(f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}")
```

### Example 4: MLflow Integration

```python
import machinegnostics as mg

# Initialize MLflow tracker
tracker = mg.integrations.MLflowTracker(
    experiment_name='bearing_diagnostics',
    tracking_uri='http://localhost:5000'
)

# Log parameters and metrics
with tracker.run():
    # Train model
    model = mg.model.RandomForest()
    model.fit(X_train, y_train)
    
    # Log results
    tracker.log_params({'n_estimators': 100})
    tracker.log_metric('accuracy', accuracy)
    tracker.log_model(model)
```

---

## Import Depth Constraints

### ✅ ALLOWED (Depth ≤ 2)
```python
from mg.magnet import Dense
from mg.magnet.activations import sech_fidelity
from mg.magcal import Characteristics
from mg.metrics import accuracy_score
from mg.data import load_diagnostic_data
```

### ❌ DISCOURAGED (Depth > 2)
```python
# Avoid deep nesting
from mg.magnet.layers.dense import Dense  # Too deep
from mg.magcal.gdf.base_df import BaseDF  # Too deep
```

### Why?
- **Shallow imports** = easier for users to remember and type
- **Cleaner IDE autocomplete** = better developer experience
- **Less cognitive load** = more intuitive API
- **Easier to refactor** = internal structure changes don't break imports

---

## Public API Design Rules

### Rule 1: All Public Classes/Functions in `__init__.py`
Every public class and function must be explicitly imported and listed in `__all__` of its package's `__init__.py`.

```python
# ✅ CORRECT
# In magnet/__init__.py
from .layers import Dense
from .activations import Fi
__all__ = ["Dense", "Fi"]

# Users can now do:
from mg.magnet import Dense, Fi

# ❌ WRONG
# Relying on users finding deep imports
from mg.magnet.layers.dense import Dense  # Discouraged
```

### Rule 2: One Import per Concept
Group related items in the same module's `__init__.py`.

```python
# ✅ CORRECT
# In magnet/losses/__init__.py
from .infidelity import InfidelityLoss
from .fidelity import FidelityLoss
__all__ = ["InfidelityLoss", "FidelityLoss"]

# Then in magnet/__init__.py
from .losses import InfidelityLoss, FidelityLoss
```

### Rule 3: No Import Side Effects
Imports should not execute expensive operations or modify global state.

```python
# ❌ WRONG
# magnet/__init__.py
from .core import initialize_gpu  # Bad: runs initialization on import
initialize_gpu()

# ✅ CORRECT
# Users call it explicitly
from mg.magnet.core import initialize_gpu
initialize_gpu()
```

### Rule 4: Clear Naming for Imports
Use descriptive names that make the origin obvious.

```python
# ✅ CLEAR
from mg.magnet import Fi  # Obviously MAGNET-specific
from mg.metrics import fidelity_score  # Obviously a metric

# ❌ UNCLEAR
from mg.magnet import Layer1  # What's Layer1?
from mg.metrics import score  # Which score?
```

---

## Module Organization by Subsystem

### MAGNET (`mg.magnet`)
**Export from**: `magnet/__init__.py`

Primary exports:
- `Dense` - Standard dense layer
- `Fi` - Fidelity layer (iDense)
- `Hi` - Irrelevance layer
- `Sequential` - Sequential model
- `InfidelityLoss`, `FidelityLoss` - Gnostic loss functions
- `sech_fidelity`, `tanh_irrelevance` - Gnostic activations

Sub-exports (depth 2):
- `mg.magnet.activations` - Activation functions
- `mg.magnet.losses` - Loss functions
- `mg.magnet.layers` - Layer implementations
- `mg.magnet.core` - Core runtime

### Magcal (`mg.magcal`)
**Export from**: `magcal/__init__.py`

Primary exports:
- `Characteristics` - Sensor characteristics
- `CalibrationModel` - Calibration algorithm
- `LayerParam` - Layer parameters
- `convert_sensor_data` - Data conversion function

Sub-exports (depth 2):
- `mg.magcal.data_conversion` - Data transformation utilities
- `mg.magcal.criteria_eval` - Evaluation criteria
- `mg.magcal.util` - Utility functions

### Model (`mg.model`)
**Export from**: `model/__init__.py`

Primary exports:
- `RandomForest` - Random forest classifier
- `SVM` - Support vector machine
- `KMeans` - K-means clustering
- `GradientBoosting` - Gradient boosting

Sub-exports (depth 2):
- `mg.model.classifiers` - Classification models
- `mg.model.regressors` - Regression models
- `mg.model.ensemble` - Ensemble methods

### Metrics (`mg.metrics`)
**Export from**: `metrics/__init__.py`

Primary exports:
- `accuracy_score` - Accuracy metric
- `precision_score` - Precision metric
- `fidelity_score` - Gnostic fidelity metric
- `mean_squared_error` - MSE metric
- `confusion_matrix` - Confusion matrix

Sub-exports (depth 2):
- `mg.metrics.diagnostic_metrics` - Diagnostic-specific metrics
- `mg.metrics.performance_metrics` - Performance evaluation

### Data (`mg.data`)
**Export from**: `data/__init__.py`

Primary exports:
- `load_diagnostic_data` - Load benchmark datasets
- `preprocess` - Data preprocessing
- `split_train_test` - Train/test splitting
- `normalize` - Data normalization

Sub-exports (depth 2):
- `mg.data.loaders` - Data loading utilities
- `mg.data.preprocessing` - Preprocessing functions

### Integrations (`mg.integrations`)
**Export from**: `integrations/__init__.py`

Primary exports:
- `MLflowTracker` - MLflow integration
- `CloudAdapter` - Cloud platform adapter
- `DeploymentManager` - Model deployment

Sub-exports (depth 2):
- `mg.integrations.mlflow_config` - MLflow configuration
- `mg.integrations.cloud_adapters` - Cloud adapters (AWS, Azure, GCP)
- `mg.integrations.deployment` - Deployment utilities

---

## Best Practices for Developers

### ✅ DO:
1. **Always export in `__init__.py`** before importing at depth 2
2. **Use explicit imports** instead of `from module import *`
3. **Keep submodule imports shallow** (depth ≤ 2)
4. **Document what's exported** in module docstrings
5. **Test imports** in your test suite

### ❌ DON'T:
1. **Don't import deep** (depth > 2) in production code
2. **Don't rely on relative imports** for users
3. **Don't use internal implementations** directly
4. **Don't skip `__all__` definition** in `__init__.py`
5. **Don't create circular imports** between subsystems

---

## Testing Imports

All agents should verify imports work correctly:

```python
# tests/test_imports.py
def test_imports_depth_0():
    """Test top-level import works"""
    import machinegnostics as mg
    assert hasattr(mg, 'magnet')
    assert hasattr(mg, 'magcal')
    assert hasattr(mg, 'model')

def test_imports_depth_1():
    """Test depth-1 imports work"""
    from mg.magnet import Dense, Fi
    from mg.magcal import Characteristics
    from mg.metrics import accuracy_score
    # All should import successfully

def test_imports_consistency():
    """Test that depth-1 imports are in __all__"""
    from mg.magnet import __all__
    assert "Dense" in __all__
    assert "Fi" in __all__
```

---

## Documentation for Users

Include in API documentation:

```markdown
## Quick Import Reference

Machine Gnostics provides intuitive imports at exactly 2 levels of depth:

### Standard Imports
\`\`\`python
import machinegnostics as mg

# MAGNET framework
from mg.magnet import Dense, Fi, Sequential
from mg.magnet.losses import InfidelityLoss

# Calibration
from mg.magcal import Characteristics

# Models
from mg.model import RandomForest

# Metrics
from mg.metrics import accuracy_score

# Data
from mg.data import load_diagnostic_data

# Integrations
from mg.integrations import MLflowTracker
\`\`\`

### Direct Access
\`\`\`python
# After importing subsystems
from mg import magnet, metrics

magnet.Dense(64)
metrics.accuracy_score(y_true, y_pred)
\`\`\`
```

---

## Import Guidelines Checklist (Manager Agent)

- [ ] All public classes in `__all__`
- [ ] All public functions in `__all__`
- [ ] Import depth ≤ 2 for user-facing code
- [ ] No internal imports exposed at depth 2
- [ ] Circular imports avoided
- [ ] Imports tested in test suite
- [ ] Documentation includes import examples
- [ ] Type hints compatible with import paths
- [ ] IDE autocomplete works for major classes/functions

