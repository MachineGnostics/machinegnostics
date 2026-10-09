# Models (Machine Gnostics MAGNET)

The `machinegnostics.magnet` module provides model containers and neural network architectures for building deep learning pipelines on gnostic algebra.

---

## Key Model Classes

<div class="grid cards" markdown>

-   **[Model](models/model.md)** — Abstract base container with `compile()`, `fit()`, `predict()`, `evaluate()`, `summary()`, and weight management APIs
-   **[Sequential](models/sequential.md)** — Linear stack of layers with straightforward forward propagation
-   **[GnosticNeuron](models/gnostic_neuron.md)** — MAGNET‑specific neuron implementing gnostic algebra transformations (`fi`, `fj`, `hi`, `hj`, `ei`)

</div>

---

## Model Utilities

- `get_model(name)` — Resolve model name to instance
- `load_model(filepath)` — Load a saved model from disk
- `save_model(model, filepath)` — Persist a model to disk

---

## Callbacks & History

- **Callback** — Base callback class for training hooks
- **EarlyStopping** — Stop training when monitored metric stops improving
- **History** — Automatically tracks epoch-level metrics during training

---

## Key Concepts

### Model Lifecycle

1. **Construction** — Define layer architecture
2. **Compilation** — Configure loss and optimizer via `compile()`
3. **Training** — Fit on data via `fit(X, y, epochs, batch_size, callbacks)`
4. **Evaluation** — Validate performance via `evaluate(X, y)`
5. **Inference** — Generate predictions via `predict(X)`

### Gnostic Parameter Handling

MAGNET models provide specialized handling for gnostic parameters:
- `s` — Scale parameter for coordinate transformation
- `z0` — Center parameter for coordinate origin
- `*_s`, `*_z0` — Layer-specific scale/center variants

Optimizers automatically apply adaptive scaling to these parameters during training.

---

**Author:** Nirmal Parmar
