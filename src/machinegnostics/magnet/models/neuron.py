"""Gnostic neuron convenience layer for MAGNET.

- Typical gnostic neuron supposed to have a single Dense layer with x inputs and 1 output, followed by a gnostic activation (fidelity).

Developer note
-------------
Author: Nirmal Parmar

Examples
--------
>>> import numpy as np
>>> from machinegnostics.magnet.models import GnosticNeuron
>>> neuron = GnosticNeuron(2, 1, activation="sigmoid")
>>> neuron(np.array([[0., 1.]])).shape
(1, 1)
"""

from __future__ import annotations

from ..activations import get_activation
from ..layers.dense import Dense
from .model import Sequential


class GnosticNeuron(Dense):
	"""Dense layer followed by a configurable activation.

	Use this for the tutorial-style logic-gate and binary classification
	examples where a single gnostic neuron is enough.
	"""

	def __init__(self, in_features, out_features=1, activation="fidelity", **kwargs):
		"""Create a gnostic neuron.

		Parameters
		----------
		in_features:
			Number of input features.
		out_features:
			Number of output units, usually 1 for binary tasks.
		activation:
			Activation name or instance, resolved through ``get_activation``.
		**kwargs:
			Additional keyword arguments forwarded to ``Dense``.
		"""
		super().__init__(in_features, out_features, **kwargs)
		self.activation = get_activation(activation)

	def forward(self, x, training=True):
		"""Apply the dense transform and then the chosen activation."""
		output = super().forward(x, training=training)
		return self.activation(output, training=training) if self.activation is not None else output
