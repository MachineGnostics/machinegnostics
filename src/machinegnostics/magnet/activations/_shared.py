"""Shared activation helpers for MAGNET.

Developer note
--------------
This module keeps the public activation implementation split across one class
per file while centralizing the gnostic tensor bridge and shared utilities.
"""

from __future__ import annotations

import numpy as np
import torch

from ..core._gnostic import compute_characteristics, custom_tensor
from ..core.tensor import Tensor
from ..layers.base import Layer


def _gnostic_activation_tensor(x, value, prime):
	prime = np.asarray(prime, dtype=np.float64)
	return custom_tensor(value, x, prime)
