import numpy as np
from typing import Tuple, List


def make_linnerud_check_data(return_names: bool = True):
    """
    Loads a Linnerud-like multi-output regression dataset.

    Size: 20 observations. 3 exercise variables and 3 physiological variables.

    Why it's famous:
    Great for multi-output regression. Commonly used to demonstrate models that
    predict multiple targets simultaneously (e.g., `Weight`, `Waist`, `Pulse`)
    from exercise features (`Chins`, `Situps`, `Jumps`).

    Returns
    -------
    X : numpy.ndarray
        Exercise features of shape (20, 3): columns ['Chins', 'Situps', 'Jumps'].
    Y : numpy.ndarray
        Physiological targets of shape (20, 3): columns ['Weight', 'Waist', 'Pulse'].
    X_names : list[str]
        Returned if `return_names=True`.
    Y_names : list[str]
        Returned if `return_names=True`.

    Notes
    -----
    - If scikit-learn is available (a project dependency), this function will
      return the exact Linnerud dataset used by scikit-learn.
    - Otherwise, it falls back to a synthetic but similarly shaped dataset.

    Example
    -------
    >>> from machinegnostics.data import make_linnerud_check_data
    >>> X, Y, Xn, Yn = make_linnerud_check_data()
    >>> X.shape, Y.shape, Xn, Yn
    ((20, 3), (20, 3), ['Chins', 'Situps', 'Jumps'], ['Weight', 'Waist', 'Pulse'])
    """
    X_names = ['Chins', 'Situps', 'Jumps']
    Y_names = ['Weight', 'Waist', 'Pulse']

    try:
        # Preferred: exact well-known dataset
        from sklearn.datasets import load_linnerud  # type: ignore
        d = load_linnerud()
        X = d.data.astype(float)
        Y = d.target.astype(float)
        if return_names:
            return X, Y, X_names, Y_names
        return X, Y
    except Exception:
        # Fallback synthetic generator with similar correlations
        rng = np.random.default_rng(42)
        n = 20
        chins = np.clip(rng.normal(10, 4, n), 0, None)
        situps = np.clip(rng.normal(160, 30, n), 60, 300)
        jumps = np.clip(rng.normal(50, 8, n), 20, 80)
        X = np.column_stack([chins, situps, jumps])

        # Physiological stats: larger exercise levels loosely imply lower weight/waist
        weight = np.clip(90 - 0.6 * chins - 0.05 * situps + rng.normal(0, 3, n), 45, 120)
        waist = np.clip(85 - 0.5 * chins - 0.06 * situps + rng.normal(0, 2.5, n), 55, 120)
        pulse = np.clip(70 + 0.15 * jumps - 0.2 * chins + rng.normal(0, 3, n), 45, 110)
        Y = np.column_stack([weight, waist, pulse])

        if return_names:
            return X.astype(float), Y.astype(float), X_names, Y_names
        return X.astype(float), Y.astype(float)
