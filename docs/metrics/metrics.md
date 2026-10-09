# Metrics (Machine Gnostics MAGNET)

The `machinegnostics.metrics` module provides evaluation functions for machine learning and data science tasks, including both classical statistical metrics and MAGNET-specific gnostic metrics.

---

## Key Metric Categories

<div class="grid cards" markdown>

-   **[Classical Metrics](metrics.md#classical)** — Accuracy, Precision, Recall, F1 Score, Confusion Matrix, Classification Report, R2, Silhouette Score
-   **[Gnostic Metrics](metrics.md#gnostic)** — Gnostic Mean, Median, Std, Variance, MAE, MSE, RMSE, R2, Relevance, DivI, GMMFE, Entropy, EvalMet, Stationarity, Auto-Correlation, Auto-Covariance, Cross-Covariance

</div>

---

## Classical Metrics

Classical statistics-based mathematics (mean, variance, accuracy, F1, R2) for evaluating model performance on standard data.

- **[Accuracy](accuracy.md)** — Classification correctness rate
- **[Precision](precision.md)** — Positive predictive value
- **[Recall](recall.md)** — Sensitivity / true positive rate
- **[F1 Score](f1_score.md)** — Harmonic mean of precision and recall
- **[Confusion Matrix](confusion_matrix.md)** — Classification error breakdown
- **[Classification Report](classification_report.md)** — Per-class precision, recall, F1
- **[R2 Score](r2.md)** — Coefficient of determination (regression)
- **[Silhouette Score](silhouette_score.md)** — Cluster separation quality

---

## Gnostic Metrics

Gnostic algebra mathematics (`fi`/`fj` fidelity/infidelity, `hi`/`hj` relevance/irrelevance, `ei` entropy, centered coordinate `θ`) applied directly to metric computation for concept-level evaluation.

- **[Gnostic Mean](g_mean.md)** — Central tendency using gnostic weights
- **[Gnostic Median](g_median.md)** — Robust center estimate
- **[Gnostic Std](g_std.md)** — Gnostic-weighted dispersion
- **[Gnostic Variance](g_variance.md)** — Weighted variance measure
- **[Gnostic MAE](mae.md)** — Mean absolute error
- **[Gnostic MSE](mse.md)** — Mean squared error
- **[Gnostic RMSE](rmse.md)** — Root mean squared error
- **[Gnostic R2](r2_score.md)** — Gnostic coefficient of determination
- **[Gnostic Relevance](g_relevance.md)** — Relevance (`hi`) based evaluation
- **[Gnostic DivI](divI.md)** — Divergence of irrelevance
- **[Gnostic GMMFE](gmmfe.md)** — Gnostic mixture model feature evaluation
- **[Gnostic Entropy](entropy.md)** — Entropy (`ei`) measurement
- **[Gnostic EvalMet](evalmet.md)** — Gnostic evaluation metric
- **[Gnostic Stationarity](stationarity_test.md)** — Stationarity test
- **[Gnostic Auto-Correlation](g_auto_corelation.md)** — Gnostic auto-correlation
- **[Gnostic Auto-Covariance](g_auto_covariance.md)** — Gnostic auto-covariance
- **[Gnostic Cross-Covariance](g_cross_variance.md)** — Cross-covariance measure

---

**Author:** Nirmal Parmar
