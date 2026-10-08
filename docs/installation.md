# Installation Guide

Machine Gnostics is distributed as a standard Python package. It is tested on macOS and Windows 11 with Python 3.11+ and is fully compatible with NumPy, pandas, SciPy, and other standard data science libraries.

---

## 1. Create a Python Virtual Environment

Best practice: use a virtual environment to avoid dependency conflicts.

=== "macOS & Linux (python)"
    ```bash
    python3 -m venv .mg-env
    source .mg-env/bin/activate
    ```

=== "macOS & Linux (uv)"
    ```bash
    uv venv .mg-env --python 3.11
    source .mg-env/bin/activate
    ```

=== "Windows (python)"
    ```cmd
    python -m venv .mg-env
    .mg-env\Scripts\activate
    ```

=== "Windows (uv)"
    ```cmd
    uv venv .mg-env --python 3.11
    .mg-env\Scripts\activate
    ```

---

## 2. Install Machine Gnostics

=== "macOS & Linux (pip)"
    ```bash
    pip install machinegnostics
    ```
=== "macOS & Linux (uv)"
    ```bash
    uv add machinegnostics
    ```
=== "Windows (pip)"
    ```cmd
    pip install machinegnostics
    ```
=== "Windows (uv)"
    ```cmd
    uv add machinegnostics
    ```

---

## 3. Verify Installation

=== "macOS & Linux"
    ```bash
    python -c "import machinegnostics; print('imported successfully!')"
    pip show machinegnostics
    ```
=== "Windows"
    ```cmd
    python -c "import machinegnostics; print('imported successfully!')"
    pip show machinegnostics
    ```

---

## 4. Quick Usage Example

```python
import numpy as np
from machinegnostics.magcal import EGDF

data = np.array([-13.5, 0, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
egdf = EGDF()
egdf.fit(data)
egdf.plot()
print(egdf.params)
```

---

## 5. Platform and Environment

- **Operating System:** macOS and Windows 11 (tested)
- **Python Version:** 3.11 or newer
- **Dependencies:** NumPy, pandas, SciPy compatible
- **Alternative managers:** `uv` (recommended for speed); `conda` or `mamba` can also manage the environment, then install with `pip install machinegnostics` inside it

---

## 6. Troubleshooting

Always activate your virtual environment before installing or running:

=== "macOS & Linux"
    ```bash
    source .mg-env/bin/activate
    # or for conda
    conda activate myenv
    ```
=== "Windows"
    ```cmd
    .mg-env\Scripts\activate
    # or for conda
    conda activate myenv
    ```

- **Upgrade pip:** `pip install --upgrade pip`
- **Check Python version:** `python --version` (must be 3.8+)
- **Clean install:** Create a fresh environment and reinstall
- **Network errors:** Ensure a stable internet connection
- **Permission errors:** Do not use `sudo pip install`; use a virtual environment instead

---

Find the full tutorial guide [here](./tutorials/tutorials.md).

---

Machine Gnostics is designed for simplicity and reliability, making robust machine learning accessible for all Python users.
