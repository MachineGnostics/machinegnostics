# Initializers (Machine Gnostics MAGNET)

The `machinegnostics.magnet` module provides weight initialization strategies for MAGNET layers. Proper initialization is critical for gradient stability in networks using gnostic activations.

---

## Key Initializer Classes

<div class="grid cards" markdown>

-   **[GlorotUniform](initializers/glorot_uniform.md)** / **[GlorotNormal](initializers/glorot_normal.md)** — Xavier/Glorot initialization (`limit = sqrt(6 / (fan_in + fan_out))`)
-   **[HeUniform](initializers/he_uniform.md)** / **[HeNormal](initializers/he_normal.md)** — He initialization for ReLU-family activations (`√(2/fan_in)` or `√(2/fan_in)` stddev)
-   **[Normal](initializers/normal.md)** — Random normal distribution (`RandomNormal` alias)
-   **[Uniform](initializers/uniform.md)** — Uniform random distribution
-   **[Ones](initializers/ones.md)** — Constant ones (useful for biases)
-   **[Zeros](initializers/zeros.md)** — Constant zeros (standard bias init)

</div>

---

## Base Class

-   **[Initializer](initializers/initializer.md)** — Abstract base (`Initializer(ABC)`) with `__init__(dtype=np.float64)` and abstract `__call__(shape) -> np.ndarray`. Includes `normalize_shape()` and `compute_fans()` helpers.

---

## Helper Function

- `get_initializer(name_or_callable=None)` — Resolve name (`'glorot_uniform'`, `'he_normal'`, `'normal'`, etc.) or callable; defaults to `GlorotUniform()`.

---

## Compatibility Aliases

- `XavierUniform = GlorotUniform`
- `XavierNormal = GlorotNormal`
- `RandomNormal = Normal`

---

**Author:** Nirmal Parmar  
