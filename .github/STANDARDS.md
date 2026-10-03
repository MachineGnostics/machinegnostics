# Machine Gnostics Code & Documentation Standards

## Import Standards

**All imports must follow shallow hierarchy rules to ensure ease of use for end users.**

### Import Depth Constraint
Maximum import depth for user-facing code: **2 levels**

```
✅ ALLOWED:
from mg.magnet import Dense                          # Depth: 2
from mg.magnet.activations import sech_fidelity     # Depth: 2
from mg.magcal import Characteristics               # Depth: 2

❌ DISCOURAGED:
from mg.magnet.layers.dense import Dense            # Depth: 3 (too deep)
from mg.magcal.gdf.base_df import BaseDF            # Depth: 3 (too deep)
```

### Public API Organization

Every package's `__init__.py` must:
1. Import all public classes/functions from submodules
2. Define `__all__` listing all exported names
3. Include clear module docstring with examples

Example: `magnet/__init__.py`
```python
"""MAGNET: Machine Gnostic Neural Network Environment/Toolkit"""

from .layers import Dense, Flatten, BatchNorm
from .activations import Fi, Hi
from .losses import InfidelityLoss, FidelityLoss
from .core import Sequential, Model

__all__ = [
    "Dense",
    "Flatten",
    "BatchNorm",
    "Fi",
    "Hi",
    "sech_fidelity",
    "tanh_irrelevance",
    "InfidelityLoss",
    "FidelityLoss",
    "Sequential",
    "Model",
]
```

### Import Organization Within Files
```python
"""
Module: <module_name>
Description: Clear description of what this module/file does.

Part of Machine Gnostics: <subsystem> (magcal, magnet, metrics, ml_models, integrations)
Author: Nirmal Parmar
Date: Current Date
License: GNU3

Key Components:
  - <Main class/function 1>: Brief description
  - <Main class/function 2>: Brief description
"""

# Future imports (always first)
from __future__ import annotations

# Standard library imports
import logging
from dataclasses import dataclass
from typing import Any, Optional

# Third-party imports
import numpy as np
from sklearn.preprocessing import StandardScaler

# Machine Gnostics imports (use shallow paths)
from machinegnostics.magcal import Characteristics
from machinegnostics.metrics import accuracy_score
from machinegnostics.magnet import Dense, Fi

logger = logging.getLogger(__name__)
```

### Import Guidelines Checklist
- [ ] All public classes/functions in module's `__all__`
- [ ] Import depth ≤ 2 for user-facing code
- [ ] Internal imports not exposed publicly
- [ ] No circular imports between subsystems
- [ ] Imports organized: future, stdlib, third-party, local
- [ ] Import statements use full paths from package root
- [ ] No wildcard imports (`from module import *`)

---

## File Header Standard

```python
class DiagnosticModel:
    """
    Brief one-line description of the class.
    
    Extended description explaining the purpose, usage, and key features.
    Include any important notes about assumptions or constraints.
    
    This class is part of the machine gnostics diagnostic framework
    and provides [specific capability].
    
    Attributes:
        param1 (str): Description of what param1 represents
        param2 (int): Description of what param2 represents
        calibration_factor (float): Scaling factor for calibration (default: 1.0)
    
    Raises:
        ValueError: When input data is invalid or out of range
        TypeError: When wrong data type is provided
    
    Example:
        >>> model = DiagnosticModel(param1="config", param2=100)
        >>> result = model.diagnose(sensor_data)
        >>> print(f"Diagnosis: {result}")
    
    Note:
        This class requires properly calibrated sensor data.
        See calibration module for details.
    
    See Also:
        CalibrationModule: For calibrating sensor data
        MetricsModule: For evaluating model performance
    """
    
    def __init__(self, param1: str, param2: int) -> None:
        """Initialize the diagnostic model with parameters."""
        self.param1 = param1
        self.param2 = param2
```

### Class Documentation Checklist
- [ ] One-line summary
- [ ] Extended description with purpose
- [ ] Attributes section with types and descriptions
- [ ] Raises section documenting exceptions
- [ ] Example usage provided
- [ ] Important notes/warnings included
- [ ] Cross-references to related modules

---

## Public Method Documentation Standard

```python
def calculate_diagnostic_score(
    sensor_data: np.ndarray,
    calibration_params: dict[str, float],
    threshold: float = 0.5,
) -> float:
    """
    Calculate the diagnostic score from sensor measurements.
    
    This method combines multiple sensor readings with calibration
    parameters to produce a normalized diagnostic score between 0 and 1,
    where higher values indicate worse machine condition.
    
    Args:
        sensor_data (np.ndarray): Input sensor measurements of shape (N,).
            Must be in units of [Tesla] for magnetic sensors.
        calibration_params (dict[str, float]): Dictionary of calibration
            factors. Required keys: 'offset', 'scale', 'temperature_comp'.
        threshold (float, optional): Decision threshold for diagnostic
            classification (default: 0.5). Range: [0, 1].
    
    Returns:
        float: Diagnostic score in range [0, 1]. Values > threshold
            indicate potential machine fault.
    
    Raises:
        ValueError: If sensor_data is empty or contains NaN values.
        ValueError: If any calibration_params values are negative.
        TypeError: If sensor_data is not a numpy array.
    
    Example:
        >>> import numpy as np
        >>> data = np.array([0.5, 0.45, 0.52, 0.48])
        >>> params = {'offset': 0.0, 'scale': 1.0, 'temperature_comp': 0.01}
        >>> score = calculate_diagnostic_score(data, params)
        >>> print(f"Score: {score:.3f}")
        Score: 0.497
    
    Note:
        Sensor data should be preprocessed and noise-filtered before
        passing to this function. See data_conversion module.
    
    Performance:
        Typical execution time: < 1ms for N < 10000 samples.
    """
    # Implementation
    pass
```

### Method Documentation Checklist
- [ ] One-line summary describing what it does
- [ ] Extended description explaining the algorithm/logic
- [ ] Args section with types, descriptions, and units
- [ ] Returns section with type and description
- [ ] Raises section documenting all exceptions
- [ ] Example showing typical usage
- [ ] Important notes about prerequisites or assumptions
- [ ] Performance characteristics if relevant

---

## Class-level Attributes Documentation

```python
@dataclass
class CalibrationConfig:
    """
    Configuration for magnetic calibration procedures.
    
    Attributes:
        calibration_mode (str): Type of calibration to perform.
            Options: 'full', 'quick', 'fine'. Default: 'full'.
        temperature_range (tuple[float, float]): Valid temperature
            range for calibration in Celsius. Default: (20, 30).
        max_error_tolerance (float): Maximum allowable calibration
            error in Tesla. Default: 0.001.
        iterations (int): Number of calibration iterations to perform.
            Default: 10. Must be > 0.
        verbose (bool): Print calibration progress. Default: False.
    
    Example:
        >>> config = CalibrationConfig(
        ...     calibration_mode='quick',
        ...     temperature_range=(15, 35),
        ...     max_error_tolerance=0.002,
        ...     verbose=True
        ... )
    """
    calibration_mode: str = "full"
    temperature_range: tuple[float, float] = (20.0, 30.0)
    max_error_tolerance: float = 0.001
    iterations: int = 10
    verbose: bool = False
```

---

## Property Documentation

```python
@property
def is_calibrated(self) -> bool:
    """
    Check if the model has been calibrated.
    
    Returns:
        bool: True if calibration has been completed, False otherwise.
    
    Example:
        >>> model = DiagnosticModel()
        >>> print(model.is_calibrated)
        False
        >>> model.calibrate(sensor_data)
        >>> print(model.is_calibrated)
        True
    """
    return self._calibration_complete
```

---

## Module-level Documentation

```python
"""
Module: magcal.gdf.base_df
Description: Base classes for generalized distribution functions.

Part of Machine Gnostics: magcal
Author: Machine Gnostics Team
Date: 2024-01-15

This module provides abstract base classes for implementing various
generalized distribution functions used in magnetic diagnostic
analysis. It includes:

Classes:
    BaseDF: Abstract base for all distribution functions
    DFCalculator: Computational engine for DF calculations

Key Concepts:
    - Distribution functions map sensor values to probability distributions
    - Used for anomaly detection and fault classification
    - Calibration required for accurate predictions

Dependencies:
    - NumPy for numerical operations
    - magcal.calibration for calibration factors
    - machinegnostics.metrics for evaluation metrics

Usage Example:
    >>> from machinegnostics.magcal.gdf import create_custom_df
    >>> df = create_custom_df(sensor_data, calibration_params)
    >>> probabilities = df.calculate_probabilities(new_data)

See Also:
    machinegnostics.magcal.characteristics: Sensor characteristics
    machinegnostics.metrics: Evaluation metrics
    machinegnostics.magnet: Neural network implementations
"""
```

---

## Test File Standards

```python
"""
Module: tests.test_magcal_calibration
Description: Unit tests for magcal calibration module.

Test Coverage:
  - CalibrationModel initialization and validation
  - Calibration algorithm accuracy
  - Error handling and edge cases
  - Integration with sensor data pipelines

Author: Machine Gnostics Team
Date: 2024-01-15
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from machinegnostics.magcal import CalibrationModel
from machinegnostics.magcal.exceptions import CalibrationError


class TestCalibrationModel:
    """Test suite for CalibrationModel class."""
    
    @pytest.fixture
    def sample_data(self):
        """Provide sample calibration data for testing."""
        return np.array([0.1, 0.15, 0.12, 0.14, 0.13])
    
    @pytest.fixture
    def calibration_config(self):
        """Provide standard calibration configuration."""
        return {
            'offset': 0.0,
            'scale': 1.0,
            'iterations': 5
        }
    
    def test_initialization_with_valid_config(self, calibration_config):
        """Test that CalibrationModel initializes correctly with valid config."""
        model = CalibrationModel(**calibration_config)
        assert model.offset == 0.0
        assert model.scale == 1.0
    
    def test_calibrate_with_valid_data(self, sample_data, calibration_config):
        """Test calibration with valid sensor data."""
        model = CalibrationModel(**calibration_config)
        result = model.calibrate(sample_data)
        
        assert result is not None
        assert isinstance(result, dict)
        assert 'calibration_factor' in result
    
    def test_calibrate_raises_error_with_empty_data(self, calibration_config):
        """Test that calibration raises error with empty data."""
        model = CalibrationModel(**calibration_config)
        with pytest.raises(ValueError, match="Data cannot be empty"):
            model.calibrate(np.array([]))
    
    def test_calibrate_raises_error_with_nan_values(self, calibration_config):
        """Test that calibration raises error with NaN values."""
        model = CalibrationModel(**calibration_config)
        data_with_nan = np.array([0.1, np.nan, 0.12])
        with pytest.raises(ValueError, match="Data contains NaN values"):
            model.calibrate(data_with_nan)
```

### Test File Checklist
- [ ] Module docstring with test coverage summary
- [ ] Imports organized and complete
- [ ] Test classes organized by functionality
- [ ] Descriptive test method names
- [ ] Fixtures for common test data
- [ ] Clear assertions with meaningful messages
- [ ] Edge cases tested
- [ ] Error conditions tested
- [ ] Integration tests included

---

## Documentation File Standards

```markdown
# Module Name: Magcal Calibration

## Overview
Brief description of what this module does and why it matters.

## Installation & Setup
How to install and configure this module.

## Quick Start
Minimal example to get started in 5 minutes.

## Concepts
Theoretical background and key concepts.

## API Reference

### Classes

#### `CalibrationModel`
Description and usage examples.

##### Methods
- `calibrate(data)`: Full method documentation
- `validate()`: Full method documentation

### Functions

#### `estimate_calibration_factor(data)`
Full documentation with examples.

## Advanced Usage
Complex scenarios and best practices.

## Examples
Complete working examples.

## Troubleshooting
Common issues and solutions.

## See Also
Links to related modules and documentation.

## References
Academic papers, technical standards, etc.
```

---

## Python File Organization Standard

**Each public class or module must reside in a dedicated Python file.**

### File-to-Class Mapping Rule

```
✅ CORRECT:
machinegnostics/magnet/layers/dense.py          → Contains Dense class
machinegnostics/magnet/layers/flatten.py        → Contains Flatten class
machinegnostics/magnet/activations/sech.py      → Contains sech_fidelity function
machinegnostics/magcal/characteristics.py       → Contains Characteristics class

❌ INCORRECT:
machinegnostics/magnet/layers.py                → Contains Dense + Flatten + Conv
machinegnostics/magcal/all_classes.py           → Contains Characteristics + Methods + Helpers
```

### Why Dedicated Files?

1. **Maintainability**: One class per file makes code easier to find and modify
2. **Testing**: Dedicated test files map 1:1 to source files
3. **Reusability**: Clear module boundaries prevent accidental coupling
4. **Documentation**: Each class gets its own dedicated documentation file
5. **Import Clarity**: Import path clearly indicates what you're getting

### Implementation Rules

1. **Public Class** → Dedicated `.py` file named after the class (lowercase with underscores)
   ```python
   class DenseLayer:
       pass
   
   # File: dense_layer.py (or dense.py if shorter)
   ```

2. **Public Function Module** → Dedicated `.py` file for related functions
   ```python
   def sech_fidelity(x):
       """Fidelity function using hyperbolic secant."""
       pass
   
   def tanh_irrelevance(x):
       """Irrelevance function using hyperbolic tangent."""
       pass
   
   # File: activations.py (groups related activation functions)
   ```

3. **Helper/Private Classes** → Can coexist with public class IF:
   - They are strictly internal (prefixed with `_`)
   - They directly support the public class
   - They are NOT imported elsewhere
   
   ```python
   # File: dense.py
   class Dense:  # Public
       pass
   
   class _DenseHelper:  # Private helper - OK to coexist
       pass
   ```

4. **Module Initialization** → Use `__init__.py` to expose public API only
   ```python
   # File: magnet/layers/__init__.py
   from .dense import Dense
   from .flatten import Flatten
   from .batch_norm import BatchNorm
   
   __all__ = ["Dense", "Flatten", "BatchNorm"]
   ```

### File Organization Checklist

- [ ] Each public class is in its own `.py` file
- [ ] Each public module groups only related functions
- [ ] File name matches class/module purpose (lowercase)
- [ ] `__init__.py` exposes only public API
- [ ] Private helpers (prefixed with `_`) are in same file as public class
- [ ] No "utils.py" or "helpers.py" with mixed unrelated functions
- [ ] Imports are clean and follow depth-2 constraint
- [ ] Related documentation file exists (e.g., docs/magnet/dense.md)

### Example File Structure

```
machinegnostics/magnet/
├── __init__.py                    # Exposes: Dense, Fi, Hi, Sequential, etc.
├── core/
│   ├── __init__.py               # Exposes: Sequential, Model
│   ├── sequential.py             # Contains Sequential class
│   └── model.py                  # Contains Model base class
├── layers/
│   ├── __init__.py               # Exposes: Dense, Flatten, BatchNorm, etc.
│   ├── dense.py                  # Contains Dense class + _DenseHelper
│   ├── flatten.py                # Contains Flatten class
│   └── batch_norm.py             # Contains BatchNorm class
├── activations/
│   ├── __init__.py               # Exposes: sech_fidelity, tanh_irrelevance
│   └── hyperbolic.py             # Contains both activation functions
└── losses/
    ├── __init__.py               # Exposes: InfidelityLoss, FidelityLoss
    ├── infidelity.py             # Contains InfidelityLoss class
    └── fidelity.py               # Contains FidelityLoss class
```

---

## Docstring Checklist (Manager Agent Validation)

For every new file:
- [ ] Module-level docstring present
- [ ] Author and date documented
- [ ] Machine Gnostics subsystem identified
- [ ] Key components listed

For every public class:
- [ ] Summary line
- [ ] Extended description
- [ ] Attributes section
- [ ] Raises section (if applicable)
- [ ] Example usage
- [ ] Related modules documented

For every public method:
- [ ] Summary line
- [ ] Extended description
- [ ] Args section with types and units
- [ ] Returns section with type
- [ ] Raises section (if applicable)
- [ ] Example usage
- [ ] Performance notes (if relevant)

---

## Code Quality Checklist (Manager Agent)

- [ ] No unused imports
- [ ] Type hints on all public APIs
- [ ] No TODO comments without context
- [ ] Error handling is explicit
- [ ] Logging statements present for important operations
- [ ] No code duplication (DRY principle)
- [ ] Functions have single responsibility
- [ ] No hardcoded magic numbers (use constants)
- [ ] Constants named in UPPER_CASE
- [ ] Consistent naming conventions

---

## Cross-module Integration Checklist (Manager Agent)

- [ ] Imports from other magcal modules are explicit
- [ ] External module dependencies documented
- [ ] API contracts clearly defined
- [ ] No circular dependencies
- [ ] Version compatibility noted for external libraries
- [ ] Integration points documented
- [ ] Error boundaries clearly defined

---

## Pre-commit Verification (Manager Agent)

Before marking code complete:
1. [ ] All docstrings are present and complete
2. [ ] Type hints are consistent
3. [ ] No broken imports
4. [ ] Code runs without warnings
5. [ ] Tests pass
6. [ ] Documentation is updated
7. [ ] Related files are consistent
8. [ ] Git diff is clean (no unrelated changes)

