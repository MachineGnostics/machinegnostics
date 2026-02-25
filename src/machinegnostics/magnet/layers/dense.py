"""Dense and Linear layers for feed-forward ANN models.

Author: Nirmal Parmar
"""

import numpy as np

from .base_layer import Layer, Parameter
from ..w_init import get_initializer


class Dense(Layer):
    """Fully connected affine layer: ``y = xW + b``."""

    def __init__(self, input_dim, output_dim, bias=True, initializer="xavier_uniform"):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.use_bias = bool(bias)

        init_fn = get_initializer(initializer)
        weight_data = init_fn((self.input_dim, self.output_dim), fan_in=self.input_dim, fan_out=self.output_dim)
        self.w = Parameter(weight_data, name="weight")

        self.b = Parameter(np.zeros((self.output_dim,), dtype=np.float64), name="bias") if self.use_bias else None

    def forward(self, inputs):
        output = inputs @ self.w
        if self.b is not None:
            output = output + self.b
        return output


class Linear(Dense):
    """Alias for Dense layer for PyTorch-style naming preference."""

    pass