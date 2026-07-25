"""Gnostic iDense layer.

This layer applies gnostic weights based on the square of fidelity (fi^2).
"""

from __future__ import annotations

import numpy as np

from ..tensor import Tensor, _compute_gnostic_bundle
from .dense import Dense


class iDense(Dense):
    """Dense layer that multiplies activations by fi squared gnostic weights."""

    def __init__(self, *args, scale_param: str | float = "auto", reference: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale_param = scale_param
        self.reference = reference

    def forward(self, inputs: Tensor, training: bool = True) -> Tensor:
        self.logger.debug(f"Forward pass through iDense layer '{self.name}'.")
        output = inputs.matmul(self.kernel)
        if self.use_bias and self.bias is not None:
            output = output + self.bias

        bundle = _compute_gnostic_bundle(output.data, scale_param=self.scale_param, reference=self.reference)
        gnostic_weights = np.asarray(bundle["fi"] ** 2, dtype=np.float64)
        self.gnostic_weights = gnostic_weights
        self.gnostic_characteristics = bundle

        weighted_output = output * Tensor(gnostic_weights, requires_grad=False)
        return self.activation(weighted_output)
