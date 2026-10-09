# Losses (Machine Gnostics MAGNET)

The losses module provides objective functions for training MAGNET models. It includes standard regression and classification losses (MSE, BinaryCrossEntropy) alongside dedicated gnostic losses (Fidelity, Infidelity, Residual Entropy, Information, RSS) that measure alignment with learned concept manifolds.

---

## Key Loss Classes

<div class="grid cards" markdown>

-   **[MSE](losses/mse.md)** — Mean squared error for regression targets
-   **[BinaryCrossEntropy](losses/binary_cross_entropy.md)** — Binary classification loss with numerical stability
-   **[FidelityLoss](losses/fidelity_loss.md)** — Maximize Mean Gnostic Fidelity
-   **[InfidelityLoss](losses/infidelity_loss.md)** — Minimize Mean Gnostic Infidelity
-   **[RSSLoss](losses/rss_loss.md)** — Relevance Squared Sum Loss
-   **[ISSLoss](losses/iss_loss.md)** — Irrelavance Squared Sum Loss
-   **[ResidualEntropyLoss](losses/residual_entropy_loss.md)** — Mean Gnostic Residual Entropy of the Data
-   **[InformationLoss](losses/information_loss.md)** — Mean Gnostic Information Measure

</div>

---

## Helper Functions

- `get_loss(name_or_obj)` — Resolve string (`'mse'`, `'fidelity'`, `'rss'`, etc.) or instance into a `Loss`.
- `fidelity_loss(y_pred, y_true, S=1)` — Convenience wrapper for `FidelityLoss`.
- `infidelity_loss(y_pred, y_true, S=1)` — Convenience wrapper for `InfidelityLoss`.
- `gnostic_weighted_mse(y_pred, y_true)` — Alias for MSE.
- `gnostic_weighted_rmse(y_pred, y_true)` — Root MSE.

---

## Why Magnet Losses?

- Standard losses provide familiar training objectives.
- Gnostic losses reward predictions that align with learned concept centers (`z0`) and scales (`S`), promoting transparent, geometry-aware optimization.
- Custom autograd kernels (`FidelityLossFunction`) provide efficient gradient computation through MAGNET tensors.

---

**Author:** Nirmal Parmar  
