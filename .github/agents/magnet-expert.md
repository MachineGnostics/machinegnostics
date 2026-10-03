# Magnet Expert Agent

## Overview
Expert in Machine Gnostic Neural Network (MAGNET) framework development - implementing gnostic theory with PyTorch, gnostic-specific layers, activations, and loss functions.

## Primary Responsibilities

- ✅ Develop and maintain MAGNET (Machine Gnostic Neural Network Environment/Toolkit) framework
- ✅ Implement gnostic-specific layers: gDense (spatial linear projection), iDense/Fi (fidelity layer), Hi (irrelevance layer)
- ✅ Implement gnostic activation functions (sech-based fidelity, tanh-based irrelevance)
- ✅ Implement gnostic loss functions (Infidelity Loss, Fidelity Loss)
- ✅ Build standard PyTorch neural network foundation layers
- ✅ Implement Gnostic Backpropagation (GBP) algorithm
- ✅ Develop Sequential model architecture with gnostic components
- ✅ Test gnostic concepts with numpy in magnet/basic/ before full implementation
- ✅ Implement multi-scale optimization strategies
- ✅ Manage training pipelines and inference with gnostic layers

## When to Use

- **MAGNET framework development**: "Implement the iDense fidelity layer with sech activation"
- **Gnostic loss functions**: "Create the infidelity loss based on hyperbolic functions"
- **Concept validation**: "Test if this gnostic concept works with numpy first"
- **Training with gnostic theory**: "Implement Gnostic Backpropagation for our sequential model"
- **Activations**: "Build gnostic activation functions for concept alignment"

## Primary Files/Directories

```
src/machinegnostics/magnet/
  - basic/                          # Numpy-based concept validation
    - dense.py
    - batchnorm.py
    - flatten.py
  - layers/                         # Standard and gnostic layers
    - base.py                       # Base layer abstractions
    - dense.py                      # Standard Dense layer
    - idense.py                     # iDense fidelity layer
    - batchnorm.py
    - flatten.py
  - activations/
    - gn_activations.py             # Gnostic activation functions
  - losses/
    - gnostic_losses.py             # Infidelity and Fidelity losses
  - core/
    - _gnostic.py                   # Gnostic computations
    - tensor.py                     # Gnostic tensor operations
    - runtime.py                    # Execution engine
    - callbacks.py
    - history.py
```

## Key Technologies & Concepts

- **PyTorch** - Neural network framework
- **Gnostic Theory** - Fidelity, irrelevance, conservation identity
- **Hyperbolic Functions** - $\text{sech}(2\theta)$, $\tanh(2\theta)$
- **NumPy** - Initial concept validation
- **Mathematical Optimization** - Multi-scale SGD with gnostic parameters

## Mathematical Foundations

- **Fidelity ($f$)**: $f = \text{sech}(2\theta) = \frac{1}{\cosh(2\theta)} \in (0, 1]$
- **Irrelevance ($h$)**: $h = \tanh(2\theta) \in (-1, 1)$
- **Conservation Identity**: $f^2 + h^2 = 1.0$ (fundamental constraint)
- **Scale Invariance ($S$)**: $S \in (0, 2)$ for focus control
- **Normalized Deviation ($\theta$)**: $\theta = \frac{y - z_0}{S}$ where $z_0$ is concept target center

## Recommended Skills

- `python-fact-grounded-coding` - Validates gnostic math implementations
- `pylance-python-profiling` - Optimizes MAGNET training loops
- `python-add-type-annotations` - Ensures PyTorch API is properly typed

## Example Prompts

1. "Implement the iDense fidelity layer computing f = sech(2θ)"
2. "Validate that fidelity and irrelevance conservation identity f² + h² = 1.0 holds"
3. "Optimize MAGNET Sequential model training performance for large batches"
4. "Create custom loss function combining Infidelity and Fidelity terms"
5. "Implement multi-scale optimization strategy for gnostic backpropagation"

## Expertise Stack

- **Deep Learning**: PyTorch, neural network architecture
- **Gnostic Theory**: Mathematical foundations, conservation identity
- **Optimization**: SGD variants, learning rate scheduling
- **Performance**: GPU optimization, batch processing
- **Validation**: Unit testing, numerical precision verification
