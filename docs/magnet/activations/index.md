# Activations (Machine Gnostics MAGNET)

The `machinegnostics.magnet` module provides activation layers for the MAGNET neural network framework. It includes standard nonlinearities (ReLU, Sigmoid, Tanh, Softmax, Step, ELU, LeakyReLU, Softplus, Swish, Square) and dedicated gnostic activation functions (Fi, Fj, Hi, Hj, Ei) that implement fidelity and information characteristics.

---

## Key Activation Classes

<div class="grid cards" markdown>

-   **[ReLU](relu.md)** — Rectified linear unit (`max(0, x)`)
-   **[Sigmoid](sigmoid.md)** — Logistic sigmoid (`1 / (1 + exp(-x))`)
-   **[Tanh](tanh.md)** — Hyperbolic tangent (zero-centered, `[-1, 1]`)
-   **[Softmax](softmax.md)** — Normalized probability distribution across last axis
-   **[Step](step.md)** — Hard threshold activation (`threshold=0.0`)
-   **[LeakyReLU](leakyrelu.md)** — Leaky rectified linear (`alpha=0.01`)
-   **[ELU](elu.md)** — Exponential linear unit (`alpha=1.0`)
-   **[Softplus](softplus.md)** — Smooth ReLU approximation (`log(1 + exp(x))`)
-   **[Swish](swish.md)** — Self-gated nonlinearity (`x * sigmoid(x)`)
-   **[Square](square.md)** — Elementwise square (`x ** 2`)
-   **[Fi](fi.md)** — Gnostic fidelity activation (`sech(2θ)` with learnable `z0`, `S`)
-   **[Fj](fj.md)** — Gnostic infidelity / complement activation
-   **[Hi](hi.md)** — Gnostic information / hyperbolic tangent activation
-   **[Hj](hj.md)** — Gnostic information complement
-   **[Ei](ei.md)** — Gnostic entropy / error activation

</div>

---

## Helper Functions

- `get_activation(name_or_obj, verbose=False)` — Resolve a string, callable, or layer instance into an activation layer.
- `ActivationFunctions` — Container with staticmethod access to Fi, Fj, Hi, Hj, Ei.

---

## Why Magnet Activations?

- Standard layers provide familiar drop-in replacements for conventional networks.
- Gnostic activations (Fi, Hi, etc.) learn concept centers (`z0`) and scales (`S`) to align with data geometry rather than fixed functions.
- All implementations preserve autograd through MAGNET's `Tensor` facade backed by PyTorch.

---

**Author:** Nirmal Parmar
