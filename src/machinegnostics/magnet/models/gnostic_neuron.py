"""Gnostic neuron convenience layer for MAGNET.

Developer note
--------------
Author: Nirmal Parmar

This module keeps the neuron convenience wrapper in its own file so the model
package has one dedicated class per module.
"""

from __future__ import annotations

from ..activations import get_activation
from ..layers.dense import Dense


class GnosticNeuron(Dense):
	"""Dense layer followed by a configurable activation.

	A gnostic neuron combines a dense projection with a configurable
	activation, making it a compact building block for binary-classification,
	logic-gate, and gnostic characteristic workflows.
	"""

	def __init__(self, in_features, out_features=1, activation="fi", verbose: bool = False, **kwargs):
		"""Create a gnostic neuron."""
		super().__init__(in_features, out_features, verbose=verbose, **kwargs)
		self.activation = get_activation(activation, verbose=verbose)

	def forward(self, x, training=True):
		"""Apply the dense transform and then the chosen activation."""
		output = super().forward(x, training=training)
		return self.activation(output, training=training) if self.activation is not None else output
