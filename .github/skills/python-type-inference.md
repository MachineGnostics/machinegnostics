# python-type-inference Skill

## Overview
Infers and validates the single correct type for specific Python symbols including function returns, parameters, variables, and class fields.

## Purpose

This skill enables:
- Determining correct types for complex expressions
- Validating type consistency in data pipelines
- Understanding complex numpy/pandas type systems
- Validating PyTorch tensor types
- Documenting inferred types

## Use This Skill When

✅ **Understanding complex types**
  - Complex numpy array types with shapes
  - Pandas DataFrame column types
  - PyTorch tensor shapes and dtypes
  - Union/Optional type resolution

✅ **Validating data pipelines**
  - Ensuring data type consistency across magcal
  - Checking PyTorch tensor types in MAGNET
  - Validating metric computation types

✅ **Debugging type mismatches**
  - Type incompatibility between modules
  - Unexpected type transformations
  - Shape mismatches in neural networks

✅ **Documenting inferred types**
  - Creating type documentation for users
  - Documenting numpy array shapes
  - PyTorch tensor type specifications

## Primary Agents

- 📊 **Magcal Expert** - Validates complex numpy/pandas types
- 🧠 **Magnet Expert** - Validates PyTorch tensor types
- 📈 **Metrics Agent** - Understands metric type flows
- 🤖 **ML Models Specialist** - Validates model I/O types

## Example Prompts

### For Magcal Expert (Data Type Validation)
```
"Use python-type-inference to analyze sensor data flow in magcal.

Determine the correct types for:
1. Raw sensor input: What numpy array dtype/shape?
2. After calibration: What dtype/shape after transformation?
3. Before metrics: What format fed to metrics computation?
4. Expected output: What type returned to users?

Verify:
- Type consistency across conversion steps
- Shape preservation where expected
- Precision loss (float64 → float32)?
- Any type incompatibilities?"
```

### For Magnet Expert (Tensor Type Validation)
```
"Use python-type-inference to analyze tensor types in Dense layer.

Determine exact types for:
1. Input tensor: Shape and dtype requirements
2. Weight matrix: Storage format and dtype
3. Bias vector: Shape and dtype
4. Output tensor: Guaranteed shape and dtype
5. Gradient tensors: For backpropagation

Validate:
- All operations preserve expected dtypes
- Batch processing maintains shape invariants
- GPU/CPU consistency
- No unexpected type conversions"
```

### For Metrics Agent (Metric Output Types)
```
"Use python-type-inference to determine return types for metrics.

For accuracy_score function, determine:
1. Input types: np.ndarray, torch.Tensor, or both?
2. Expected shapes: (N,)? (N, 1)?
3. Return type: float? np.float64? torch.Tensor?
4. Edge cases: Empty arrays? Single sample?

Document:
- Exact return type specification
- Type coercion if needed
- Any automatic conversions
- Type validation examples"
```

### For ML Models Specialist (Model I/O Types)
```
"Use python-type-inference to validate model types.

Infer types for Random Forest model:
1. Expected input: (n_samples, n_features)? dtype?
2. Return type: np.ndarray? shape (n_samples,)?
3. Prediction probabilities: shape (n_samples, n_classes)?
4. Can model accept different dtypes?

Verify:
- Consistency with scikit-learn spec
- Type validation in practice
- Edge cases: Single sample? Large batch?
- GPU compatibility if applicable"
```

## Type Inference Examples

### NumPy Arrays
```python
# Raw sensor data
sensor_data: np.ndarray  # dtype: float32, shape: (batch_size, n_sensors)

# After calibration
calibrated: np.ndarray  # dtype: float32, shape: (batch_size, n_sensors)

# Metric computation
metric_input: np.ndarray  # dtype: float64, shape: (batch_size, n_features)
```

### PyTorch Tensors
```python
# Dense layer input
x: torch.Tensor  # dtype: torch.float32, shape: (batch_size, input_size)

# Weight matrix
W: torch.Tensor  # dtype: torch.float32, shape: (input_size, output_size)

# Output
output: torch.Tensor  # dtype: torch.float32, shape: (batch_size, output_size)
```

### Complex Types
```python
# Union type
def process(data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    # Input can be either type
    # Always returns tensor
    pass

# Optional type
def train(model: nn.Module, callbacks: Optional[List[Callback]] = None):
    # callbacks can be None or List
    pass

# Tuple types
def get_data() -> Tuple[np.ndarray, np.ndarray]:
    # Returns (features, labels)
    pass
```

## Type Validation Patterns

### Data Pipeline Type Flow
```python
# Step 1: Load raw data
raw: np.ndarray  # float32, (n, 64)

# Step 2: Calibrate
calibrated: np.ndarray  # float32, (n, 64)

# Step 3: Convert for metrics
metric_input: torch.Tensor  # float32, (n, 64)

# Step 4: Compute
metric_output: torch.Tensor  # float32, (n,)

# Type consistency check:
# - All numeric operations preserve precision ✓
# - Shapes preserved through pipeline ✓
# - No unexpected conversions ✓
```

### Model I/O Types
```python
# Input validation
assert isinstance(input_data, (np.ndarray, torch.Tensor))
assert input_data.shape[1] == expected_features
assert input_data.dtype in [np.float32, np.float64]

# Output validation
output = model.predict(input_data)
assert isinstance(output, np.ndarray)
assert output.shape == (input_data.shape[0],)
assert output.dtype == np.float64
```

## Success Criteria

✅ Exact types determined for all symbols  
✅ Type consistency validated  
✅ Shape invariants verified  
✅ No unexpected type conversions  
✅ Documentation generated from inferred types  

## Recommended Workflow

1. **Identify symbol** - Which type to infer?
2. **Trace flow** - How does data flow through code?
3. **Infer type** - Use skill to determine exact type
4. **Validate** - Check against runtime behavior
5. **Document** - Record inferred types for users

## Related Skills

- `python-add-type-annotations` - For annotating inferred types
- `python-fact-grounded-coding` - For validating type correctness
- `pylance-python-profiling` - For type-based optimization
