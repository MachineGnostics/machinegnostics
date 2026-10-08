# Magnet: Machine Gnostic Neural Network

**MAGNET** is the deep-learning framework of the Machine Gnostics library. It provides a deterministic, finite, and algebraic foundation for building neural networks—an alternative to traditional probabilistic backpropagation approaches.

!!! tip "NOTE"
    We are actively developing additional MAGNET neural network layers, models, and training methods. Stay tuned for updates as new tools and documentation become available.

    We are open to collaboration and new ideas. If you’re interested in contributing, sharing feedback, or exploring partnerships in neural network design, feel free to connect with us—your insights and creativity are always welcome!

---

## Key Magnet Modules

<div class="grid cards" markdown>

-   **[Activations](activations.md)**

    Activation layers including standard nonlinearities (ReLU, Sigmoid, Tanh, Softmax) and dedicated gnostic activations (Fi, Fj, Hi, Hj, Ei) that implement fidelity and information characteristics.

-   **[Losses](losses.md)**

    Loss functions for training, including standard MSE and binary cross-entropy, plus dedicated gnostic losses: Fidelity, Infidelity, Residual Entropy, Information, and RSS Loss.

-   **[Initializers](initializers.md)**

    Weight initialization strategies (Glorot/Xavier, He, Normal, Uniform, Zeros, Ones) to stabilize training and gradient flow.

-   **[Optimizers](optimizers.md)**

    Optimization algorithms (SGD, Adam, RMSprop, Adagrad) with adaptive special-parameter scaling for gnostic layers.

-   **[Models](models.md)**

    Model containers (Sequential, Model) and the GnosticNeuron component for constructing and training networks.

-   **[Layers](layers.md)**

    Layer implementations including Dense, BatchNorm, GnosticBatchNorm, Flatten, iDense, and jDense.

-   **[Core](core.md)**

    Core infrastructure: Tensor, History, Callback, EarlyStopping, and runtime configuration.

</div>

---

## Why Use Magnet?

- **Deterministic & Finite:** All computations are reproducible and free from randomness.
- **Event-Level Modeling:** Uncertainty and error are handled at the level of individual data events.
- **Algebraic Inference:** Leverages gnostic algebra and error geometry for transparent, explainable results.
- **Resilient Architecture:** Designed to withstand outliers, corrupted data, and distributional shifts.
- **PyTorch-Backed:** Built on PyTorch tensors under the hood. MAGNET is fully torch-compatible — gradients flow through PyTorch autograd automatically, and you can mix MAGNET layers with standard torch.nn modules when needed. The public API stays in MAGNET terms while leveraging PyTorch's efficient GPU acceleration and optimization.

---

## Quick Start

```python
from machinegnostics.magnet import Sequential, Dense, ReLU, MSE, Adam
import numpy as np

model = Sequential(layers=[
    Dense(in_features=4, out_features=8),
    ReLU(),
    Dense(in_features=8, out_features=1)
])
model.compile(loss=MSE(), optimizer=Adam())
model.fit(np.random.rand(100, 4), np.random.rand(100, 1), epochs=5)
```

---

## Next Steps

- Browse individual module pages for detailed class documentation.
- Try hands-on notebooks and examples in the documentation examples folder.
- Check [Magnet Concepts](../magnet/index.md) for the architectural vision and collaboration information.

---

**Author:** Nirmal Parmar  
