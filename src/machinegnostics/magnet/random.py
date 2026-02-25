"""Random seed controls for reproducible Magnet experiments.

Author: Nirmal Parmar
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Set deterministic seed across Python and NumPy RNGs."""
    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    os.environ["PYTHONHASHSEED"] = str(value)


__all__ = ["set_seed"]