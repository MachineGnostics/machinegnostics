# Glossary

!!! abstract
    This document provides definitions and explanations for the main arguments and variables used in **Machine Gnostics** data analytics, machine learning, and deep learning models. Understanding these concepts will help users grasp the unique characteristics of the Machine Gnostics library, which is based on the non-statistical paradigm of **Mathematical Gnostics**.

---

## Core Concepts

Machine Gnostics
:   A machine learning and deep learning library founded on Mathematical Gnostics, a non-statistical paradigm for data analysis.

Mathematical Gnostics
:   An alternative to traditional statistical methods, focusing on the quantification and estimation of uncertainty in data.

---

## Key Arguments and Gnostic Characteristics

### 1. Gnostic Geometries

These are the fundamental variables used to describe data in the gnostic framework. They are divided into two main spaces:

- **Quantifying Space (Q-space, \( j \)):** Describes the variability and irrelevance in the data.
- **Estimating Space (E-space, \( i \)):** Describes the estimation of variability and relevance.

**Comparative Table of Geometries:**

| Feature | Quantifying (Q-space) | Estimating (E-space) |
| :--- | :--- | :--- |
| **Variability** | **\( f_j \)** (Quantifying data variability) | **\( f_i \)** (Estimating data variability) |
| **Relevance/Irrelevance** | **\( h_j \)** (Quantifying irrelevance/error) | **\( h_i \)** (Estimating relevance) |
| **Probability** | **\( p_j \)** (Quantifying probability) | **\( p_i \)** (Estimating probability) |
| **Information** | **\( I_j \)** (Quantifying information) | **\( I_i \)** (Estimating information) |
| **Entropy** | **\( e_j \)** (Quantifying entropy) | **\( e_i \)** (Estimating entropy) |

All four key variables (\( f_j, h_j, f_i, h_i \)) are collectively called **gnostic characteristics**.

---

### Detailed Definitions

#### 2. Entropy & Residuals

- **\( e_i \)**: *Estimating entropy*  
  Entropy estimate for the data.

- **\( e_j \)**: *Quantifying entropy*  
  Quantifies the entropy content.

- **\( re \)**: *Residual entropy*  
  The remaining entropy after estimation, representing the difference between quantification and estimation entropy.

#### 3. Loss Functions

- **\( H_c \) loss**: *Gnostic mean relevance loss*  
  A loss function based on gnostic relevance, where \( c \) can be \( i \) or \( j \).

---

!!! tip "Further Reading"
    For more detailed mathematical background, see the foundational texts on [References](../ref/references.md) and the documentation of the Machine Gnostics library.