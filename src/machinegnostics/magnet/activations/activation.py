"""Dedicated public activation module for activation.

Developer note
--------------
This module keeps the public MAGNET activation API one class per file.
"""

from __future__ import annotations

import numpy as np
import torch

from ._shared import Tensor, compute_characteristics, custom_tensor, _gnostic_activation_tensor, Layer

class Activation(Layer):
	"""Base class for MAGNET activation layers.

	This class defines the shared interface for all standard and gnostic
	activations in MAGNET. Subclasses implement ``forward`` and return a tensor
	that participates in the library's autograd flow.

	Examples
	--------
	>>> class DoubleActivation(Activation):
	... 	def forward(self, x, training=True):
	... 		return x if isinstance(x, Tensor) else Tensor(x)
	>>> isinstance(DoubleActivation(), Activation)
	True
	"""
	def __init__(self, name=None, verbose: bool = False):
		super().__init__(name, verbose=verbose)

	def forward(self, x, training=True):
		"""Transform the input tensor and return the activated output."""
		raise NotImplementedError

	def backward(self, grad_output):
		"""Activation layers use tensor autograd, so explicit backward is unused."""
		raise NotImplementedError
