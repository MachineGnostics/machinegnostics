# python-add-type-annotations Skill

## Overview
Adds inline type annotations to Python source to make code type-safe, improve IDE support, and enhance documentation for users and developers.

## Purpose

This skill adds or validates type annotations for:
- Function parameters and return types
- Variable and class attributes
- Complex data structures
- Generic types and unions

## Use This Skill When

✅ **Publishing public API**
  - All exported classes need type annotations
  - Public methods need parameter and return types
  - Improves IDE autocomplete and support

✅ **Improving IDE support**
  - Adding types enables better error detection
  - Enables refactoring support
  - Improves documentation hover info

✅ **Validating type consistency**
  - Ensuring types match across modules
  - Checking compatibility between subsystems
  - Documenting complex types

✅ **Creating type documentation**
  - Documenting PyTorch tensor types
  - Showing expected data shapes
  - Clarifying complex type unions

## Primary Agents

- 🧠 **Magnet Expert** - Types PyTorch layer APIs
- 📊 **Magcal Expert** - Types data transformation functions
- 📈 **Metrics Agent** - Types diagnostic metric functions
- 🤖 **ML Models Specialist** - Types model interfaces
- 🎯 **Manager Agent** - Validates type consistency
- 📚 **Documentation Agent** - Documents complex types
- ☁️ **Integration Specialist** - Types cloud SDK interactions

## Example Prompts

### For Magnet Expert (PyTorch Types)
```
"Add type annotations to all public classes in magnet/ module.

Specifically annotate:
1. Dense layer: input shape → output shape
2. Fi layer: fidelity tensor types
3. Hi layer: irrelevance tensor types
4. Sequential model: batch processing types
5. Loss functions: input types, output scalar"
```

### For All Agents
```
"Ensure all public classes/functions have complete type annotations:

1. Function parameters: 'def method(data: ndarray, scale: float) -> Tensor:'
2. Return types: Always include for public APIs
3. Attributes: 'self.weights: torch.Tensor'
4. Class attributes: Document types at class level
5. Complex types: Use Union, Optional, List, Dict as needed"
```

### For Manager Agent (Consistency Check)
```
"Validate that all public APIs across magnet/, magcal/, metrics/ have:
1. Complete type annotations
2. Consistent type naming across modules
3. Proper use of Optional/Union for nullable types
4. Clear documentation of complex type unions"
```

### For Documentation Agent
```
"Add type annotations to all examples and ensure:
1. Parameter types clearly shown
2. Return types documented
3. Data shape expectations documented
4. Type Union alternatives explained"
```

## Annotation Best Practices

### Function Annotations
```python
from typing import List, Optional, Union
import torch
import numpy as np

def process_data(
    data: Union[np.ndarray, torch.Tensor],
    scale: float = 1.0,
    name: Optional[str] = None
) -> torch.Tensor:
    """Process data and return tensor."""
    pass
```

### Class Annotations
```python
class Fi:
    """Fidelity layer using gnostic theory."""
    
    # Class attributes
    fidelity_fn: Callable[[torch.Tensor], torch.Tensor]
    scale_param: float
    
    # Instance initialization
    def __init__(
        self,
        input_size: int,
        scale: float = 1.0
    ) -> None:
        self.input_size = input_size
        self.scale_param = scale
```

### Complex Type Examples
```python
# Data loading
def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (features, labels)."""
    pass

# Model output
def predict(
    data: torch.Tensor
) -> Dict[str, Union[torch.Tensor, float]]:
    """Return dict with predictions and confidence."""
    pass

# Optional parameters
def train(
    model: nn.Module,
    epochs: int,
    callbacks: Optional[List[Callback]] = None
) -> None:
    """Train model with optional callbacks."""
    pass
```

## Success Criteria

✅ All public APIs fully type-annotated  
✅ IDE support enhanced (autocomplete works)  
✅ Type consistency across modules  
✅ Complex types properly documented  
✅ No type annotation errors  

## Recommended Workflow

1. **Identify scope** - Which classes/functions to annotate?
2. **Check dependencies** - Are third-party types available?
3. **Run validation** - Use skill to add annotations
4. **Test IDE support** - Verify autocomplete works
5. **Document** - Update docstrings with type info

## Related Skills

- `python-type-inference` - For complex type validation
- `pylance-docs` - For documenting types in docstrings
- `python-fact-grounded-coding` - For validating type correctness
