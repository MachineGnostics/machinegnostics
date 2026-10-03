# pylance-refactoring Skill

## Overview
Automates code refactoring for consistent patterns, import organization, and standards enforcement across the codebase.

## Purpose

This skill enables:
- Automated refactoring with workspace consistency
- Import organization and depth enforcement
- Wildcard import removal
- Unused import cleanup
- __all__ definition verification

## Use This Skill When

✅ **Enforcing import standards**
  - Verify all imports comply with max-depth-2 constraint
  - Check that __all__ is defined in __init__.py files
  - Ensure consistent import patterns

✅ **Code quality cleanup**
  - Remove unused imports workspace-wide
  - Convert wildcard imports to explicit imports
  - Enforce naming conventions
  - Organize code structure

✅ **Standards compliance**
  - Ensure consistent formatting
  - Apply code quality rules
  - Maintain project standards

## Primary Agents

- 🎯 **Manager Agent** - Enforces standards, cleans code
- 🧯 **All Agents** (periodic) - For code quality maintenance

## Example Prompts

### For Manager Agent (Standards Enforcement)
```
"Use pylance-refactoring to ensure import standards compliance.

Specifically:
1. Identify any imports deeper than depth-2 in src/machinegnostics/
2. Verify __all__ is defined in:
   - magnet/__init__.py
   - magcal/__init__.py
   - metrics/__init__.py
   - ml_models/__init__.py
   - integrations/__init__.py
3. Remove all unused imports
4. Convert wildcard imports to explicit imports
5. Report all files needing fixes"
```

### For Workspace Cleanup
```
"Use pylance-refactoring to clean the entire workspace:

1. Remove unused imports from all Python files
2. Ensure consistent import organization
3. Verify no depth-3+ imports are exposed
4. Check __all__ definitions
5. Report summary of changes"
```

### For Specific Module
```
"Use pylance-refactoring on the magnet module:

1. Verify all public classes exported in __init__.py
2. Remove unused internal imports
3. Check that depth-2 constraint is respected
4. Ensure consistent import ordering
5. Report detailed findings"
```

## Refactoring Patterns

### Import Depth Enforcement
```python
# Before (violates depth-2 constraint)
from mg.magnet.layers.dense import Dense

# After (complies with depth-2)
from mg.magnet import Dense
```

### Wildcard Import Conversion
```python
# Before
from mg.magnet import *

# After
from mg.magnet import Dense, Fi, Hi, Sequential
```

### Unused Import Removal
```python
# Before
import numpy as np
import pandas as pd  # Unused
import torch

def process(data):
    return torch.tensor(data)

# After
import torch

def process(data):
    return torch.tensor(data)
```

### __all__ Definition Verification
```python
# In magnet/__init__.py
from .layers import Dense, Fi, Hi
from .activations import sech_fidelity, tanh_irrelevance
from .core import Sequential

__all__ = [
    "Dense",
    "Fi",
    "Hi",
    "Sequential",
    "sech_fidelity",
    "tanh_irrelevance",
]
```

## Standards to Enforce

### Import Organization
```python
# Order: future, stdlib, third-party, local
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

from machinegnostics.magnet import Dense
```

### Import Depth
```
✅ ALLOWED (depth 2):
from mg.magnet import Dense
from mg.magcal import Characteristics

❌ FORBIDDEN (depth 3+):
from mg.magnet.layers.dense import Dense
from mg.magcal.gdf.base import BaseDF
```

### Public API Definition
```python
# Every subsystem's __init__.py must have __all__
__all__ = [
    "PublicClass1",
    "PublicClass2",
    "public_function",
]
```

## Success Criteria

✅ No imports deeper than depth-2  
✅ __all__ defined in all subsystems  
✅ No unused imports  
✅ No wildcard imports  
✅ Consistent import ordering  

## Recommended Workflow

1. **Define standards** - What patterns to enforce?
2. **Run refactoring** - Use skill on specific scope
3. **Review changes** - Examine proposed refactoring
4. **Approve and apply** - Merge changes
5. **Verify** - Ensure no side effects
6. **Document** - Share new patterns with team

## Related Skills

- `python-add-type-annotations` - For cleaner code
- `pylance-docs` - For documentation quality
- `python-fact-grounded-coding` - For logic validation
