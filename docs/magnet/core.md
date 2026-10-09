# Core (Machine Gnostics MAGNET)

The `machinegnostics.magnet` module provides the fundamental infrastructure for MAGNET: the `Tensor` facade, `History`, `Callback`, `EarlyStopping`, runtime configuration, and gnostic characteristic computation.

---

## Key Core Classes

<div class="grid cards" markdown>

-   **[Tensor](core/tensor.md)** — Torch-backed tensor facade with `data`, `grad`, `shape`, arithmetic, `backward()`, `zero_grad()`
-   **[History](core/history.md)** — Training history container (`record()`, `last()`)
-   **[Callback](core/callback.md)** — Base training callback (`set_model`, `on_train_*`, `on_epoch_*`)
-   **[EarlyStopping](core/early_stopping.md)** — Stop training when monitored metric stalls (`monitor='val_loss'`, `patience=5`)

</div>

---

## Runtime Configuration

- `configure(device='auto', dtype='float64', seed=None, deterministic=True)`
- `get_runtime()`, `get_torch_device()`, `get_torch_dtype()`
- `to_numpy()`, `to_torch()` — Conversion helpers

---

## Gnostic Helpers

- `compute_characteristics()` — Compute `fi`, `fj`, `hi`, `hj` from values
- `custom_tensor()` — Bridge NumPy results into torch autograd
- `gnostic_weights_i()` / `gnostic_weights_j()` — Weight computations

---

**Author:** Nirmal Parmar  

