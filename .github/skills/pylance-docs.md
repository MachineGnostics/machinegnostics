# pylance-docs Skill

## Overview
Audits and validates Python docstrings for completeness, consistency, and adherence to documentation standards.

## Purpose

This skill enables:
- Verification of docstring presence and completeness
- Validation of docstring format consistency
- Checking for missing parameter documentation
- Ensuring Examples and Raises sections exist
- Cross-reference validation

## Use This Skill When

✅ **Creating API documentation**
  - Verify all public classes have docstrings
  - Check parameter documentation completeness
  - Ensure Examples section exists
  - Validate Raises section documents exceptions

✅ **Auditing docstring quality**
  - Workspace-wide docstring coverage check
  - Module or subsystem audit
  - Pre-release API documentation check

✅ **Maintaining documentation standards**
  - Enforce consistent format
  - Check See Also links are valid
  - Verify docstring examples are correct

✅ **Preparing for documentation generation**
  - MkDocs site generation
  - Sphinx API documentation
  - Auto-generated API reference

## Primary Agents

- 📚 **Documentation Agent** - Audits docstrings, ensures completeness
- 🎯 **Manager Agent** - Quality gates for public APIs

## Example Prompts

### For Documentation Agent (Module Audit)
```
"Use pylance-docs to audit all public classes in the magnet module.

For each public class, verify:
1. Class has complete docstring
2. One-line summary is present
3. Extended description explains purpose
4. Attributes section with types and descriptions
5. Example usage section with working code
6. Raises section documents exceptions
7. See Also section links to related classes
8. No broken cross-references

Report:
- Classes missing docstrings
- Classes missing specific sections
- Examples that need fixing
- Cross-reference issues"
```

### For Multiple Modules
```
"Use pylance-docs to audit docstring coverage across:
- src/machinegnostics/magnet/
- src/machinegnostics/magcal/
- src/machinegnostics/metrics/

Report overall completion percentage and specific gaps."
```

### For Quality Gate
```
"Before releasing v1.0, use pylance-docs to verify:
1. 100% of public classes have docstrings
2. All public methods documented
3. All parameters have type and description
4. Examples section present for key classes
5. No broken references"
```

## Docstring Standards

### Class Documentation
```python
class Fi:
    """
    Fidelity layer implementing gnostic theory.
    
    Extended description explaining the purpose, usage, and key features.
    Include any important notes about assumptions or constraints.
    
    This layer computes fidelity using f = sech(2θ) where θ = (y - z₀)/S.
    
    Attributes:
        input_size (int): Number of input features
        scale (float): Scale invariance parameter S ∈ (0, 2)
        concept_center (float): Concept target center z₀
    
    Raises:
        ValueError: If scale is not in (0, 2)
        TypeError: If inputs not torch.Tensor
    
    Example:
        >>> layer = Fi(input_size=64, scale=1.0)
        >>> x = torch.randn(32, 64)
        >>> output = layer(x)
        >>> print(output.shape)
        torch.Size([32, 1])
    
    Note:
        Conservation identity f² + h² = 1.0 is maintained.
        For numerical stability, use float32 or float64.
    
    See Also:
        Hi: Irrelevance layer
        gnostic_losses: Loss functions for gnostic training
    """
```

### Method Documentation
```python
def forward(
    self,
    x: torch.Tensor,
    scale: Optional[float] = None
) -> torch.Tensor:
    """
    Compute fidelity for input tensor.
    
    Parameters:
        x (torch.Tensor): Input tensor of shape (batch_size, input_size)
        scale (Optional[float]): Override scale parameter. If None, uses self.scale.
    
    Returns:
        torch.Tensor: Fidelity values of shape (batch_size, 1)
    
    Raises:
        ValueError: If x shape doesn't match expected input_size
        RuntimeError: If CUDA device unavailable when model on GPU
    
    Example:
        >>> layer = Fi(input_size=64)
        >>> x = torch.randn(32, 64)
        >>> fidelity = layer.forward(x)
        >>> assert fidelity.shape == (32, 1)
    """
    pass
```

### Function Documentation
```python
def accuracy_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: bool = True
) -> Union[float, int]:
    """
    Compute classification accuracy.
    
    Calculates the fraction of predictions that are correct.
    
    Parameters:
        y_true (np.ndarray): Ground truth labels
        y_pred (np.ndarray): Predicted labels
        normalize (bool): If True, return fraction. If False, return count.
    
    Returns:
        Union[float, int]: Accuracy (0-1 if normalize=True, int count if False)
    
    Raises:
        ValueError: If y_true and y_pred have different shapes
    
    Example:
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_pred = np.array([0, 1, 0, 0, 1])
        >>> accuracy = accuracy_score(y_true, y_pred)
        >>> print(f"Accuracy: {accuracy:.2%}")
        Accuracy: 80.00%
    
    Note:
        For imbalanced datasets, consider using balanced_accuracy_score.
    
    See Also:
        balanced_accuracy_score: For imbalanced data
        precision_score: For precision metric
    """
    pass
```

## Audit Checklist

For each public class/function:
- [ ] Docstring present
- [ ] One-line summary provided
- [ ] Extended description explains purpose
- [ ] Parameters documented with types
- [ ] Return type documented
- [ ] Raises section present (if raises exceptions)
- [ ] Example section provided
- [ ] Notes/warnings included if needed
- [ ] See Also links present
- [ ] No broken cross-references

## Success Criteria

✅ 100% of public APIs documented  
✅ All parameters have descriptions  
✅ Examples are complete and correct  
✅ Raises sections document exceptions  
✅ Cross-references are valid  

## Recommended Workflow

1. **Define standards** - Establish docstring template
2. **Audit** - Use skill to find gaps
3. **Fix gaps** - Documentation Agent adds missing sections
4. **Validate** - Re-run audit to verify completeness
5. **Generate** - Create API documentation
6. **Maintain** - Periodic audits during development

## Related Skills

- `python-add-type-annotations` - For type documentation
- `python-fact-grounded-coding` - For validating examples
- `pylance-refactoring` - For consistency enforcement
