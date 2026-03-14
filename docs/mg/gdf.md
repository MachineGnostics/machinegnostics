# Gnostic Distribution Functions (GDF)

!!! abstract
    **Gnostic Distribution Functions (GDF)** are a new class of probability and density estimators designed for robust, flexible, and assumption-free data analysis. Unlike traditional statistical distributions, GDFs do not require any prior assumptions about the underlying data distribution. Instead, they allow the data to "speak for themselves," making them especially powerful for small, noisy, or uncertain datasets.

## Why Use GDF?

!!! question "Key Benefits"
    - [x] **No A Priori Assumptions:** GDFs do not rely on predefined parametric forms or statistical models.
    - [x] **Robustness:** They are inherently robust to outliers and inner noise, making them suitable for real-world, contaminated, or uncertain data.
    - [x] **Flexibility:** GDFs adapt to both homogeneous and heterogeneous data samples, providing detailed insights into data structure.
    - [x] **Wide Applicability:** Useful for probability estimation, density estimation, cluster analysis, and homogeneity testing.

## Four Types of Gnostic Distribution Functions

GDFs are organized along two axes:

1.  **Local vs. Global:** Local functions use weighted irrelevance, while global functions use normalized weights.
2.  **Estimating vs. Quantifying:** Estimating functions use estimating irrelevance, while quantifying functions use quantifying irrelevance.

This results in four types:

=== "ELDF"
    **Estimating Local Distribution Function**
    
    - **Characteristic:** Outlier-resistant, highly flexible.
    - **Best for:** Cluster analysis, detailed structure exploration.

=== "EGDF"
    **Estimating Global Distribution Function**
    
    - **Characteristic:** Outlier-resistant, unique for each sample.
    - **Best for:** Homogeneity testing, robust global summaries.

=== "QLDF"
    **Quantifying Local Distribution Function**
    
    - **Characteristic:** Inlier-resistant, highly flexible.
    - **Best for:** Analysis where loose clustering or inlier noise is a factor.

=== "QGDF"
    **Quantifying Global Distribution Function**
    
    - **Characteristic:** Inlier-resistant, unique for each sample.
    - **Best for:** Robust estimation when inliers (dense clusters) dominate.

## Key Concepts

Data-Driven
:   GDFs are parameterized directly by your data, not by external assumptions.

Robustness
:   **ELDF** and **EGDF** are robust to outliers, while **QLDF** and **QGDF** are robust to inliers (dense clusters).

Flexibility
:   **Local functions** (ELDF, QLDF) are highly flexible and ideal for cluster analysis and exploring marginal data structures. **Global functions** (EGDF, QGDF) are best for homogeneity testing and robust probability estimation.

Scale Parameter
:   The flexibility of local functions is controlled by a scale parameter, allowing you to "zoom in" on data structure.

!!! success "Practical Guidance"

    - Use **local functions** for exploratory, granular analysis and cluster detection.
    - Use **global functions** for summary, sample-wide analysis and homogeneity testing.
    - Choose **estimating functions** when robustness to outliers is needed.
    - Choose **quantifying functions** when robustness to inliers is important.

!!! done "Summary"
    GDFs provide robust, flexible tools for probability and density estimation, especially in challenging data scenarios. The four types allow you to tailor your analysis to the nature of your data and the goals of your study. By removing the constraints of traditional statistical models, GDFs open new possibilities for data-driven insights.

---

For implementation details and examples, see the [Tutorials](../tutorials/tutorials.md).