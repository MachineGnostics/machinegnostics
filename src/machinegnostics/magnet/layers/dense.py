"""Fully connected layer."""

from __future__ import annotations

from typing import List, Optional

from ..activations import ActivationLike, get_activation
from ..initializers import InitializerLike, get_initializer
from ..tensor import Tensor
from .base import Layer


class Dense(Layer):
    """A fully connected neural network layer."""

    def __init__(
        self,
        units: int,
        input_dim: Optional[int] = None,
        activation: ActivationLike = None,
        use_bias: bool = True,
        kernel_initializer: InitializerLike | None = None,
        bias_initializer: InitializerLike | None = None,
        name: str | None = None,
        verbose: bool = False,
    ):
        super().__init__(name=name, verbose=verbose)
        self.units = units
        self.input_dim = input_dim
        self.activation = get_activation(activation)
        self.use_bias = use_bias
        self.kernel_initializer = get_initializer(kernel_initializer)
        self.bias_initializer = get_initializer(bias_initializer or "zeros")
        self.kernel: Tensor | None = None
        self.bias: Tensor | None = None

    @property
    def params(self) -> List[Tensor]:
        params = []
        if self.kernel is not None:
            params.append(self.kernel)
        if self.use_bias and self.bias is not None:
            params.append(self.bias)
        return params

    def build(self, input_shape):
        input_dim = self.input_dim or input_shape[-1]
        kernel = self.kernel_initializer((input_dim, self.units))
        self.kernel = Tensor(kernel, requires_grad=True)
        if self.use_bias:
            self.bias = Tensor(self.bias_initializer((self.units,)), requires_grad=True)
        self.built = True
        self.logger.info(f"Built Dense layer with input_dim={input_dim}, units={self.units}.")

    def forward(self, inputs: Tensor, training: bool = True) -> Tensor:
        self.logger.debug(f"Forward pass through Dense layer '{self.name}'.")
        output = inputs.matmul(self.kernel)
        if self.use_bias and self.bias is not None:
            output = output + self.bias
        return self.activation(output)
