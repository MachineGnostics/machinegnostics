"""Gnostic neuron convenience model for MAGNET.

Developer note
--------------
Author: Nirmal Parmar

``GnosticNeuron`` combines one dense affine transform with one activation.
Mathematically it computes ``activation(x @ W + b)`` and is useful when a full
multi-layer container would be overkill.
"""

from __future__ import annotations

from ..activations import get_activation
from ..layers.dense import Dense

from typing import Any, Iterable


class GnosticNeuron(Dense):
	"""Dense layer followed by a configurable activation.

	A gnostic neuron combines a dense projection with a configurable
	activation, making it a compact building block for binary-classification,
	logic-gate, and gnostic characteristic workflows.

	The default fidelity activation is ``"fi"``, which makes the neuron behave
	like a gnostic output unit without any extra wiring.

	This class does not add a new training loop or parameter container; it is a
	convenience subclass of ``Dense`` that only changes the forward pass by
	applying the selected activation.

	Parameters
	----------
	in_features: int
		Number of input features.
	out_features: int
		Number of output units, usually 1 for binary tasks.
	activation: str | callable
		Activation name or instance, resolved through ``get_activation``.
	verbose: bool
		Enable debug logging for the dense layer and any activation resolved
		from a string.
	**kwargs:
		Additional keyword arguments forwarded to ``Dense``.

	Attributes
	----------
	activation: callable
		The resolved activation function applied after the dense transform.
	activation_name: str
		Name of the activation function, useful for logging and debugging.

	Notes
	-----
	This class does not introduce new parameters beyond those in ``Dense``.
	The primary purpose is to combine a dense layer with a chosen activation in a single convenient unit.

	Examples
	--------
	>>> import numpy as np
	>>> from machinegnostics.magnet import GnosticNeuron
	>>> neuron = GnosticNeuron(2, 1, activation="fi")
	>>> neuron(np.array([[0.0, 1.0]])).shape
	(1, 1)
	"""

	def __init__(self, in_features: int, out_features: int = 1, activation="fi", verbose: bool = False, **kwargs) -> None:
		"""Create a gnostic neuron.

		This is a convenience wrapper around ``Dense`` followed by a chosen
		activation. It is useful when you want a single trainable layer with a
		clear activation choice.

		Parameters
		----------
		in_features: int
			Number of input features.
		out_features: int
			Number of output units, usually 1 for binary tasks.
		activation: str | callable
			Activation name or instance, resolved through ``get_activation``.
		verbose: bool
			Enable debug logging for the dense layer and any activation resolved
			from a string.
		**kwargs:
			Additional keyword arguments forwarded to ``Dense``.

		Examples
		--------
		>>> import numpy as np
		>>> from machinegnostics.magnet import GnosticNeuron
		>>> neuron = GnosticNeuron(2, 1, activation="fi")
		>>> neuron(np.array([[0.0, 1.0]])).shape
		(1, 1)
		"""
		super().__init__(in_features, out_features, verbose=verbose, **kwargs)
		self.activation = get_activation(activation, verbose=verbose)
		self.activation_name = activation if isinstance(activation, str) else activation.__class__.__name__

	def forward(self, x: Any, training: bool = True):
		"""Apply the dense transform and then the chosen activation.

		Parameters
		----------
		x: array-like or Tensor
			Input array or tensor.
		training: bool
			Training flag forwarded to the activation, kept for API consistency.

		Returns
		-------
		Any
			Output of the dense transform after the activation is applied.

		Examples
		--------
		>>> import numpy as np
		>>> from machinegnostics.magnet import GnosticNeuron
		>>> neuron = GnosticNeuron(2, 1, activation="fi")
		>>> neuron(np.array([[0.0, 1.0]])).shape
		(1, 1)
		"""
		output = super().forward(x, training=training)
		return self.activation(output, training=training) if self.activation is not None else output
