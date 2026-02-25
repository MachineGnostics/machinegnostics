"""Dropout layers for regularized ANN training.

Author: Nirmal Parmar
"""

import numpy as np

from .base_layer import Layer
from ..tensor import Tensor


class Dropout(Layer):
    """Applies inverted dropout during training and identity during eval."""

    def __init__(self, p=0.5):
        super().__init__()
        if p < 0.0 or p >= 1.0:
            raise ValueError("Dropout probability `p` must be in [0, 1).")
        self.p = float(p)

    def forward(self, inputs):
        if not self.training or self.p == 0.0:
            return inputs
        keep_prob = 1.0 - self.p
        mask = (np.random.rand(*inputs.shape) < keep_prob).astype(inputs.data.dtype) / keep_prob
        return inputs * Tensor(mask, requires_grad=False)
