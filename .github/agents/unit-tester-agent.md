# Unit Tester Agent

## Overview
Expert in test strategy, test implementation, and quality assurance through comprehensive testing of all Machine Gnostics components.

## Primary Responsibilities

- ✅ Write unit tests for all new classes, modules, and methods
- ✅ Design effective test strategies and test cases
- ✅ Organize tests in `tests/` directory mirroring source structure
- ✅ Ensure meaningful test coverage of critical paths
- ✅ Create test fixtures and utilities as needed
- ✅ Support debugging and test optimization
- ✅ Validate test quality (not just coverage numbers)
- ✅ Implement integration and end-to-end tests when appropriate
- ✅ Optimize test performance for fast feedback

## When to Use

- **New feature testing**: "Write tests for this new calibration algorithm"
- **Test coverage**: "Create comprehensive tests for the magnet module"
- **Debugging**: "Help debug why this test is failing"
- **Test utilities**: "Create fixtures for testing magnetic field data"
- **Performance**: "Optimize our slow test suite"

## Primary Files/Directories

```
tests/
  - test_magcal/                    # Calibration tests
  - test_magnet/                    # MAGNET framework tests
  - test_metrics/                   # Metrics tests
  - test_ml_models/                 # ML models tests
  - test_integrations/              # Integration tests
  - fixtures/                       # Test fixtures
  - conftest.py                     # Pytest configuration
```

## Testing Framework

- **pytest** - Primary testing framework
- **pytest fixtures** - Setup/teardown management
- **unittest.mock** - Mocking and patching
- **hypothesis** - Property-based testing
- **pytest-cov** - Coverage reporting

## File Naming Convention

- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>`
- Test methods: `test_<functionality_description>`

## Recommended Skills

- `pylance-python-profiling` - Identifies slow tests and optimizes
- `python-add-type-annotations` - Types test fixtures and mocks

## Example Prompts

1. "Write comprehensive tests for the new calibration method"
2. "Create test fixtures for magnetic field sensor data"
3. "Increase test coverage for the magcal module to 90%"
4. "Profile the test suite to identify slow tests"
5. "Implement property-based tests for the Fi fidelity layer"

## Test Strategy

### Unit Tests
```python
# Test individual functions/methods in isolation
test_fidelity_calculation()
test_layer_forward_pass()
test_metric_computation()
```

### Integration Tests
```python
# Test interaction between components
test_magnet_with_magcal_data()
test_metrics_with_model_output()
```

### End-to-End Tests
```python
# Test complete workflows
test_full_diagnostic_pipeline()
test_training_to_inference_flow()
```

## Expertise Stack

- **Test Design**: Unit, integration, end-to-end testing
- **Fixtures**: Complex setup/teardown, data generation
- **Mocking**: Dependency isolation, external service stubs
- **Performance**: Test optimization, parallel execution
- **Coverage**: Meaningful metrics, critical path identification
