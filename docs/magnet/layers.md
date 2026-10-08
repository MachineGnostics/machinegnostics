# Layers (Machine Gnostics MAGNET)

The `machinegnostics.magnet` module provides layer implementations for constructing MAGNET networks.

---

## Key Layer Classes

<div class="grid cards" markdown>

-   **[Dense](layers/dense.md)** — Fully connected affine layer (`y = x @ W + b`)
-   **[iDense](layers/idense.md)** — Gnostic estimating weight dense layer (`S`, `lel`)
-   **[jDense](layers/jdense.md)** — Gnostic quantifying weight dense layer (`S`, `lel`)
-   **[BatchNorm](layers/batchnorm.md)** — Standard batch normalization (`gamma`, `beta`, running stats)
-   **[GnosticBatchNorm](layers/gnostic_batchnorm.md)** — Batch norm with gnostic weighting (`kind="i"`/`"j"`)
-   **[Flatten](layers/flatten.md)** — Flatten structured inputs to `(batch, -1)`
-   **[Layer](layers/layer.md)** — Abstract base for MAGNET layers (`params`, `grads`, `parameters()`)

</div>

---

**Author:** Nirmal Parmar  

