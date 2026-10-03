# python-fact-grounded-coding Skill

## Overview
Grounds Python code analysis and implementation in verified facts, runtime values, diagnostics, selected interpreter state, tests, and debugger evidence before changing code or reporting conclusions.

## Purpose

This skill is essential for ensuring that code implementations are verified against:
- Mathematical specifications and formulas
- Algorithmic correctness
- Theoretical foundations
- Runtime behavior validation
- Test evidence

## Use This Skill When

✅ **Implementing gnostic theory**
  - Validating fidelity function: f = sech(2θ)
  - Validating irrelevance function: h = tanh(2θ)
  - Verifying conservation identity: f² + h² = 1.0
  - Ensuring Gnostic Backpropagation matches theory

✅ **Mathematical implementations**
  - Grounding metric formulas in statistical theory
  - Validating calibration calculations
  - Ensuring numerical precision
  - Verifying edge cases

✅ **Model validation**
  - Confirms model implementations match algorithms
  - Validates hyperparameter ranges
  - Ensures reproducibility
  - Checks numerical stability

✅ **Debugging mathematical inconsistencies**
  - Finding formula implementation errors
  - Identifying numerical precision issues
  - Validating parameter ranges

## Primary Agents

- 🧠 **Magnet Expert** - Validates gnostic math implementations
- 📊 **Magcal Expert** - Verifies calibration calculations
- 📈 **Metrics Agent** - Grounds metric formulas in theory
- 🤖 **ML Models Specialist** - Validates model implementations

## Example Prompts

### For Magnet Expert
```
"Use python-fact-grounded-coding to verify that the Fi layer 
(iDense) implementation correctly computes fidelity f = sech(2θ) 
where θ = (y - z₀)/S. 

Ground this in:
1. The mathematical definition of sech
2. The conservation identity f² + h² = 1.0
3. Test cases for various input ranges
4. Numerical precision for edge cases"
```

### For Magcal Expert
```
"Use python-fact-grounded-coding to verify that the calibration 
algorithm correctly implements the expected mathematical formula.

Validate:
1. Formula implementation matches specification
2. Numerical precision is adequate
3. Edge cases are handled correctly
4. Results match expected behavior"
```

### For Metrics Agent
```
"Use python-fact-grounded-coding to ground the accuracy_score 
metric in statistical theory.

Verify:
1. Formula: TP/(TP+FN) is correctly implemented
2. Mathematical definition matches implementation
3. Edge cases (empty predictions) handled correctly
4. Results match scikit-learn reference implementation"
```

### For ML Models Specialist
```
"Use python-fact-grounded-coding to validate that the Random Forest 
implementation matches the algorithm specification.

Check:
1. Tree building follows algorithm
2. Bootstrap sampling correct
3. Voting mechanism correct
4. Results reproducible with same seed"
```

## Success Criteria

✅ Implementation verified against mathematical specification  
✅ Test evidence provided for validation  
✅ Edge cases checked and documented  
✅ Numerical precision adequate for domain  
✅ Runtime behavior matches theory  

## Recommended Workflow

1. **State the specification** - What should the code do mathematically?
2. **Show the implementation** - What code exists?
3. **Provide test data** - What inputs/outputs to check?
4. **Run validation** - Use skill to ground analysis in facts
5. **Review findings** - Adjust code or spec as needed

## Related Skills

- `python-type-inference` - For complex type validation
- `pylance-python-profiling` - For performance validation
- `python-add-type-annotations` - For documenting verified types
