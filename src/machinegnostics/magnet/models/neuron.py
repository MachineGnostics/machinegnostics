"""Gnostic neuron convenience layer for MAGNET.

- Typical gnostic neuron supposed to have a single Dense layer with x inputs and 1 output, followed by a gnostic activation (fidelity).

Developer note
-------------
Author: Nirmal Parmar

Examples
--------
>>> import numpy as np
>>> from machinegnostics.magnet import GnosticNeuron
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

	A gnostic neuron combines a dense projection with a configurable
	activation, making it a compact building block for binary-classification,
	logic-gate, and gnostic characteristic workflows.

	The default activation is ``"fidelity"``, which makes the neuron behave
	like a gnostic output unit without any extra wiring.

	This class does not add a new training loop or parameter container; it is a
	convenience subclass of ``Dense`` that only changes the forward pass by
	applying the selected activation.
	"""

	def __init__(self, in_features, out_features=1, activation="fidelity", verbose: bool = False, **kwargs):
		"""Create a gnostic neuron.

		This is a convenience wrapper around ``Dense`` followed by a chosen
		activation. It is useful when you want a single trainable layer with a
		clear activation choice.

		Parameters
		----------
		in_features:
			Number of input features.
		out_features:
			Number of output units, usually 1 for binary tasks.
		activation:
			Activation name or instance, resolved through ``get_activation``.
		verbose:
			Enable debug logging for the dense layer and any activation resolved
			from a string.
		**kwargs:
			Additional keyword arguments forwarded to ``Dense``.
		"""
		super().__init__(in_features, out_features, verbose=verbose, **kwargs)
		self.activation = get_activation(activation, verbose=verbose)

	def forward(self, x, training=True):
		"""Apply the dense transform and then the chosen activation.

		Parameters
		----------
		x:
			Input array or tensor.
		training:
			Training flag forwarded to the activation, kept for API consistency.

		Returns
		-------
		Tensor
			Output of the dense transform after the activation is applied.

		Examples
		--------
		>>> import numpy as np
		>>> from machinegnostics.magnet import GnosticNeuron
		>>> neuron = GnosticNeuron(2, 1, activation="sigmoid")
		>>> neuron(np.array([[0.0, 1.0]])).shape
		(1, 1)
		"""
		output = super().forward(x, training=training)
		return self.activation(output, training=training) if self.activation is not None else output
