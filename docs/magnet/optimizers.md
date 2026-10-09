# Optimizers (Machine Gnostics MAGNET)

The `machinegnostics.magnet` module provides optimization algorithms for training MAGNET neural networks, with specialized handling for gnostic parameters.

---

## Key Optimizer Classes

<div class="grid cards" markdown>

-   **[Adam](optimizers/adam.md)** — Adaptive Moment Estimation with per-parameter learning rates and bias correction
-   **[SGD](optimizers/sgd.md)** — Stochastic Gradient Descent with optional momentum and Nesterov acceleration
-   **[RMSprop](optimizers/rmsprop.md)** — Root Mean Square Propagation with moving average of squared gradients
-   **[Adagrad](optimizers/adagrad.md)** — Adaptive Gradient with accumulated squared gradient scaling

</div>

---

## Base Class

-   **[Optimizer](optimizers/optimizer.md)** — Abstract base (`Optimizer(ABC)`) with `step()`, `state_dict()`, `load_state_dict()`, `zero_grad()`, `get_lr()`, `set_lr()` methods

---

## Helper Function

- `get_optimizer(name_or_callable)` — Resolve name (`'adam'`, `'sgd'`, `'rmsprop'`, `'adagrad'`) or callable; defaults to `Adam()`.

---

## Gnostic Parameter Scaling

MAGNET optimizers apply specialized scaling for gnostic parameters:
- Parameters named `s` or ending in `_s` (scale parameters)
- Parameters named `z0` or ending in `_z0` (center parameters)

These parameters receive adaptive learning rate adjustments based on their role in the gnostic coordinate transformation `θ = (x - z0) / S`.

---

## Choosing an Optimizer

| Optimizer | Best For | Key Feature |
| :--- | :--- | :--- |
| **Adam** | General-purpose, default choice | Adaptive per-parameter rates, bias correction |
| **SGD** | Convex problems, fine control | Simple, momentum support |
| **RMSprop** | Non-stationary objectives | Moving average of gradients |
| **Adagrad** | Sparse data, NLP | Accumulated gradient scaling |

---

**Author:** Nirmal Parmar
