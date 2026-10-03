# Machine Gnostics - Import Architecture Overview

## Executive Summary

Machine Gnostics provides an **intuitive, shallow import hierarchy** that keeps imports at exactly 2 levels of depth, making it easy for users to discover and use features while maintaining clean internal organization.

---

## The Import Architecture

### Top Level Access
```python
import machinegnostics as mg

# Subsystems always accessible at depth 1
mg.magnet          # MAGNET framework
mg.magcal          # Calibration module
mg.model           # ML models
mg.metrics         # Evaluation metrics
mg.data            # Data utilities
mg.integrations    # Cloud/MLflow integration
```

### Standard Imports (Depth 2)
```python
# MAGNET framework
from mg.magnet import Dense, Fi, Hi, Sequential
from mg.magnet.losses import InfidelityLoss, FidelityLoss
from mg.magnet.activations import sech_fidelity, tanh_irrelevance

# Calibration
from mg.magcal import Characteristics, CalibrationModel
from mg.magcal.data_conversion import convert_sensor_data

# Models
from mg.model import RandomForest, SVM, KMeans
from mg.model.classifiers import NeuralNetClassifier

# Metrics
from mg.metrics import accuracy_score, fidelity_score
from mg.metrics.diagnostic_metrics import bearing_fault_score

# Data
from mg.data import load_diagnostic_data, preprocess
from mg.data.loaders import load_benchmark_dataset

# Integrations
from mg.integrations import MLflowTracker, CloudAdapter
from mg.integrations.deployment import DeploymentManager
```

---

## Why This Design?

### ✅ Benefits for Users
1. **Easy Discovery**: IDE autocomplete shows all available items at depth 2
2. **Intuitive**: Logical grouping by subsystem makes imports obvious
3. **Memorable**: Only 6 subsystem names to remember (magnet, magcal, model, metrics, data, integrations)
4. **Flat API**: No buried imports requiring knowledge of internal structure
5. **Backward Compatible**: Can expand without breaking existing imports

### ✅ Benefits for Developers
1. **Organized**: Clear separation of concerns by subsystem
2. **Maintainable**: Internal refactoring doesn't break user imports
3. **Scalable**: Can add new modules within subsystems without changing API
4. **Testable**: Clear public API makes testing easier
5. **Documented**: `__all__` lists make API explicit and discoverable

### ✅ Benefits for Project
1. **Professional**: Resembles mature projects (TensorFlow, PyTorch, scikit-learn)
2. **Consistent**: All subsystems follow same pattern
3. **Predictable**: Users can guess import paths
4. **Flexible**: Easy to reorganize internals if needed

---

## Design Constraints

### Depth Limits
```
Level 0: import machinegnostics as mg
Level 1: mg.magnet, mg.magcal, mg.model, mg.metrics, mg.data, mg.integrations
Level 2: All public classes/functions available here
Level 3+: Discouraged for user-facing imports
```

### Why Maximum Depth = 2?
- **Too shallow (depth 1)**: Becomes cluttered and confusing
  - `from mg import Dense` could be ambiguous (which Dense?)
  - `from mg import accuracy_score` mixes unrelated things
  
- **Too deep (depth > 2)**: Users must know internal structure
  - `from mg.magnet.core.layers.dense import Dense` is unwieldy
  - Users must navigate directory structure to find imports
  
- **Perfect (depth 2)**: Clear, organized, discoverable
  - `from mg.magnet import Dense` is unambiguous
  - `from mg.metrics import accuracy_score` is obvious
  - IDE autocomplete shows all options at depth 2

---

## Module Organization by Subsystem

### 1. MAGNET (`mg.magnet`)
```
🟦 Level 1: mg.magnet

Level 2 Public API:
  • Dense          - Standard dense layer
  • Fi             - Fidelity layer (iDense)
  • Hi             - Irrelevance layer
  • Sequential     - Sequential model
  • Model          - Base model class
  • InfidelityLoss - Gnostic infidelity loss
  • FidelityLoss   - Gnostic fidelity loss
  • sech_fidelity  - Sech activation
  • tanh_irrelevance - Tanh activation

Level 2 Submodules:
  • mg.magnet.layers       - Layer implementations
  • mg.magnet.losses       - Loss functions
  • mg.magnet.activations  - Activation functions
  • mg.magnet.core         - Core runtime
  • mg.magnet.basic        - NumPy validation
```

### 2. Magcal (`mg.magcal`)
```
🟪 Level 1: mg.magcal

Level 2 Public API:
  • Characteristics      - Sensor characteristics
  • CalibrationModel     - Calibration algorithm
  • LayerParam           - Layer parameters
  • convert_sensor_data  - Data conversion function

Level 2 Submodules:
  • mg.magcal.data_conversion  - Data utilities
  • mg.magcal.criteria_eval    - Evaluation criteria
  • mg.magcal.gdf              - Distribution functions
  • mg.magcal.util             - Utility functions
```

### 3. Model (`mg.model`)
```
🟩 Level 1: mg.model

Level 2 Public API:
  • RandomForest        - Random forest classifier
  • SVM                 - Support vector machine
  • KMeans              - K-means clustering
  • GradientBoosting    - Gradient boosting
  • NeuralNetClassifier - Neural network classifier

Level 2 Submodules:
  • mg.model.classifiers  - Classification models
  • mg.model.regressors   - Regression models
  • mg.model.ensemble     - Ensemble methods
```

### 4. Metrics (`mg.metrics`)
```
🟨 Level 1: mg.metrics

Level 2 Public API:
  • accuracy_score        - Accuracy metric
  • precision_score       - Precision metric
  • recall_score          - Recall metric
  • fidelity_score        - Gnostic fidelity metric
  • mean_squared_error    - MSE
  • r2_score              - R² score
  • confusion_matrix      - Confusion matrix

Level 2 Submodules:
  • mg.metrics.diagnostic_metrics   - Diagnostic metrics
  • mg.metrics.performance_metrics  - Performance metrics
  • mg.metrics.validators           - Metric validators
```

### 5. Data (`mg.data`)
```
🟧 Level 1: mg.data

Level 2 Public API:
  • load_diagnostic_data  - Load benchmark datasets
  • preprocess            - Data preprocessing
  • split_train_test      - Train/test splitting
  • normalize             - Data normalization
  • validate              - Data validation

Level 2 Submodules:
  • mg.data.loaders         - Data loading utilities
  • mg.data.preprocessing   - Preprocessing functions
```

### 6. Integrations (`mg.integrations`)
```
🟥 Level 1: mg.integrations

Level 2 Public API:
  • MLflowTracker      - MLflow integration
  • CloudAdapter       - Cloud platform adapter
  • DeploymentManager  - Model deployment
  • ExperimentTracker  - Experiment tracking

Level 2 Submodules:
  • mg.integrations.mlflow_config   - MLflow configuration
  • mg.integrations.cloud_adapters  - Cloud adapters
  • mg.integrations.deployment      - Deployment utilities
```

---

## Implementation Checklist

### For Package Owners (Each Subsystem Lead)

- [ ] Create `__init__.py` in subsystem root
- [ ] Import all public classes/functions from submodules
- [ ] Define `__all__` explicitly listing exports
- [ ] Add module docstring with quick start example
- [ ] Ensure no internal imports leak to depth 2
- [ ] Test all depth-2 imports work correctly
- [ ] Document API in subsystem README
- [ ] Add imports to test suite

Example for `magnet/__init__.py`:
```python
"""
MAGNET: Machine Gnostic Neural Network Environment/Toolkit

Quick Start:
    from mg.magnet import Dense, Fi, Sequential
    model = Sequential([Dense(64), Fi(32)])
"""

# Import public API
from .layers import Dense, Flatten, BatchNorm
from .activations import Fi, Hi, sech_fidelity, tanh_irrelevance
from .losses import InfidelityLoss, FidelityLoss
from .core import Sequential, Model, GnosticRuntime

# Export public API
__all__ = [
    "Dense", "Flatten", "BatchNorm",
    "Fi", "Hi", "sech_fidelity", "tanh_irrelevance",
    "InfidelityLoss", "FidelityLoss",
    "Sequential", "Model", "GnosticRuntime",
]

# Version and metadata
__version__ = "1.0.0"
__author__ = "Machine Gnostics Team"
```

### For Top-Level Package (`machinegnostics/__init__.py`)

- [ ] Import all subsystems (magnet, magcal, model, etc.)
- [ ] Define `__all__` listing subsystems
- [ ] Optional: Re-export most-used items at top level
- [ ] Add comprehensive module docstring
- [ ] Include usage examples

Example:
```python
"""
Machine Gnostics: Framework for robust machine learning

Quick Start:
    import machinegnostics as mg
    
    # MAGNET framework
    from mg.magnet import Dense, Fi
    
    # Calibration
    from mg.magcal import Characteristics
    
    # Models
    from mg.model import RandomForest
"""

# Import subsystems
from . import magnet
from . import magcal
from . import model
from . import metrics
from . import data
from . import integrations

# Export subsystems
__all__ = [
    "magnet",
    "magcal", 
    "model",
    "metrics",
    "data",
    "integrations",
]

# Optional: convenience top-level imports
from .metrics import accuracy_score, mean_squared_error
from .data import load_diagnostic_data
```

---

## Testing Import Structure

Create `tests/test_imports.py`:

```python
"""Test that import structure works correctly"""

def test_subsystem_imports():
    """Test all subsystems accessible at depth 1"""
    import machinegnostics as mg
    assert hasattr(mg, 'magnet')
    assert hasattr(mg, 'magcal')
    assert hasattr(mg, 'model')
    assert hasattr(mg, 'metrics')
    assert hasattr(mg, 'data')
    assert hasattr(mg, 'integrations')

def test_magnet_imports():
    """Test MAGNET depth-2 imports work"""
    from mg.magnet import Dense, Fi, Hi, Sequential
    from mg.magnet.losses import InfidelityLoss
    from mg.magnet.activations import sech_fidelity
    # All should import successfully

def test_all_defined():
    """Test that __all__ is defined in all packages"""
    from mg import magnet, magcal, model, metrics, data, integrations
    for module in [magnet, magcal, model, metrics, data, integrations]:
        assert hasattr(module, '__all__')
        assert isinstance(module.__all__, list)
        assert len(module.__all__) > 0
```

---

## Documentation for Users

Include in API documentation and README:

### Import Reference
```
# Quick Import Guide

## Import Pattern
All imports follow: from mg.SUBSYSTEM import ITEM

## Available Subsystems
- magnet      # MAGNET neural network framework
- magcal      # Calibration and computational modules
- model       # Machine learning models
- metrics     # Evaluation metrics
- data        # Data utilities and loaders
- integrations # MLflow and cloud integrations

## Examples

# MAGNET framework
from mg.magnet import Dense, Fi, Sequential

# Calibration
from mg.magcal import Characteristics, CalibrationModel

# Models
from mg.model import RandomForest, SVM

# Metrics
from mg.metrics import accuracy_score, fidelity_score

# Data
from mg.data import load_diagnostic_data

# Integrations
from mg.integrations import MLflowTracker
```

---

## Migration Path (if needed)

If internal reorganization is needed:

1. **Update `__init__.py`** to expose same public API
2. **Update imports** internally (refactor submodule structure)
3. **Keep depth-2 imports unchanged** for users
4. **Deprecate any depth-3+ imports** with warnings
5. **Document migration** for early adopters

Example:
```python
# Before reorganization
from mg.magnet.core.layers.dense import Dense

# After reorganization (same external API)
from mg.magnet import Dense

# Internal structure changed but users don't notice!
```

---

## Summary

| Aspect | Design Decision |
|--------|-----------------|
| **Max Depth** | 2 levels (mg.subsystem.item) |
| **Subsystems** | 6 main: magnet, magcal, model, metrics, data, integrations |
| **Public API** | All exported via `__all__` in `__init__.py` |
| **User Imports** | Always depth-2 maximum |
| **Internal** | Can be any depth, not exposed |
| **Refactoring** | Can reorganize internally without breaking imports |
| **IDE Support** | Autocomplete works perfectly at depth 2 |

---

## References

- **IMPORT-GUIDE.md** - Detailed import implementation guide
- **STANDARDS.md** - Code standards including import organization
- **copilot-instructions.md** - Project instructions

