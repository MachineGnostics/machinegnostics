"""Standalone gnostic neuron helper."""

from __future__ import annotations

from .activations import get_activation
from .layers.dense import Dense


class GnosticNeuron(Dense):
	def __init__(self, in_features, out_features=1, activation="fidelity", **kwargs):
		super().__init__(in_features, out_features, **kwargs)
		self.activation = get_activation(activation)

	def forward(self, x, training=True):
		output = super().forward(x, training=training)
		return self.activation(output, training=training) if self.activation is not None else output
