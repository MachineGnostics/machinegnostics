"""
Dense and Linear layers for feed-forward Sequential ANN and other models.

Author: Nirmal Parmar
"""

import numpy as np

from .base_layer import Layer, Parameter
from ..w_init import get_initializer


class Dense(Layer):
    """
    Fully connected (dense) layer for neural networks.

    This layer computes an affine transformation:
        y = xW + b
    where x is the input, W is the weight matrix, and b is the bias vector.

    **Usage:**
        Use as a building block in feed-forward neural networks (MLP, classifier, regressor, etc.).
        Place between activation layers (e.g., ReLU, Sigmoid) for non-linearity.

    Parameters
    ----------
    input_dim : int
        Number of input features (columns in x).
    output_dim : int
        Number of output features (neurons in this layer).
    bias : bool, default=True
        If True, includes a bias vector b in the computation.
    initializer : str, default="xavier_uniform"
        Weight initialization method (see Magnet docs for options).

    Attributes
    ----------
    w : Parameter
        Weight matrix of shape (input_dim, output_dim).
    b : Parameter or None
        Bias vector of shape (output_dim,) if bias=True, else None.
    input_dim : int
        Number of input features.
    output_dim : int
        Number of output features.
    use_bias : bool
        Whether bias is included.

    Notes
    -----
    - The Dense layer is the most common layer in fully connected neural networks.
    - For best results, use with an activation function (e.g., ReLU, Sigmoid) after each Dense layer except the output.
    - Supports custom weight initialization via the `initializer` argument.

    Example
    -------
    >>> from machinegnostics.magnet import Dense, ReLU, Sequential
    >>> model = Sequential([
    ...     Dense(8, 32), ReLU(), Dense(32, 1)
    ... ])
    """

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