"""Gnostic jDense layer.

This layer applies the inverse of gnostic weights with clipping for stability.
"""

from __future__ import annotations

import numpy as np

from ..tensor import Tensor, _compute_gnostic_bundle
from .dense import Dense


class jDense(Dense):
    """Dense layer that multiplies activations by clipped inverse gnostic weights."""

    def __init__(self, *args, scale_param: str | float = "auto", reference: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale_param = scale_param
        self.reference = reference

    def forward(self, inputs: Tensor, training: bool = True) -> Tensor:
        self.logger.debug(f"Forward pass through jDense layer '{self.name}'.")
        output = inputs.matmul(self.kernel)
        if self.use_bias and self.bias is not None:
            output = output + self.bias

        bundle = _compute_gnostic_bundle(output.data, scale_param=self.scale_param, reference=self.reference)
        base_weights = np.clip(np.asarray(bundle["weights"], dtype=np.float64), 1e-12, 1e12)
        inverse_weights = np.clip(1.0 / base_weights, 1e-12, 1e12)
        self.gnostic_weights = inverse_weights
        self.gnostic_characteristics = bundle

        weighted_output = output * Tensor(inverse_weights, requires_grad=False)
        return self.activation(weighted_output)
