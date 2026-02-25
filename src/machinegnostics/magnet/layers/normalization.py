"""Normalization layers for ANN stability.

Author: Nirmal Parmar

Notes:
- This file currently provides `BatchNorm1d`, suitable for dense/tabular pipelines.
"""

import numpy as np

from .base_layer import Layer, Parameter
from ..tensor import Tensor


class BatchNorm1d(Layer):
    """Batch Normalization over feature dimension for 2D inputs: (batch, features)."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.affine = bool(affine)

        self.running_mean = np.zeros((1, self.num_features), dtype=np.float64)
        self.running_var = np.ones((1, self.num_features), dtype=np.float64)

        self.gamma = Parameter(np.ones((1, self.num_features), dtype=np.float64), name="gamma") if self.affine else None
        self.beta = Parameter(np.zeros((1, self.num_features), dtype=np.float64), name="beta") if self.affine else None

    def forward(self, inputs):
        if inputs.data.ndim != 2:
            raise ValueError("BatchNorm1d expects input with shape (batch_size, num_features).")
        if inputs.data.shape[1] != self.num_features:
            raise ValueError("Input feature size does not match num_features.")

        if self.training:
            batch_mean = inputs.mean(axis=0, keepdims=True)
            centered = inputs - batch_mean
            batch_var = (centered * centered).mean(axis=0, keepdims=True)

            self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * batch_mean.data
            self.running_var = (1.0 - self.momentum) * self.running_var + self.momentum * batch_var.data
        else:
            batch_mean = Tensor(self.running_mean, requires_grad=False)
            centered = inputs - batch_mean
            batch_var = Tensor(self.running_var, requires_grad=False)

        normalized = centered / (batch_var + self.eps).sqrt()

        if self.affine:
            return self.gamma * normalized + self.beta
        return normalized
