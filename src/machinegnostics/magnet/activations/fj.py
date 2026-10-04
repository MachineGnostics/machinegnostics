"""Dedicated public activation module for fj.

Developer note
--------------
This module keeps the public MAGNET activation API one class per file.
"""

from __future__ import annotations

import numpy as np
import torch

from ._shared import Tensor, compute_characteristics, custom_tensor, _gnostic_activation_tensor, Layer
from .activation import Activation
from .fi import Fi

class Fj(Activation):
	"""
	Gnostic Infidelity (Fj) Activation Layer with Learnable Center and Scale.

	This layer implements the infidelity characteristic: y_out = 1/sech(2*theta)
	where theta = (y - z0) / S. Fj is the reciprocal of the fidelity characteristic,
	emphasizing prediction uncertainty or dissimilarity from the center.

	Examples
	--------
	>>> import numpy as np
	>>> from machinegnostics.magnet import Dense, Fj, Sequential
	>>> model = Sequential([Dense(2, 1), Fj()])
	>>> model(np.array([[0.0, 1.0]])).shape
	(1, 1)
	"""

	def __init__(self, S: float | str = "auto", z0_init: float | str = "median", 
	             S_init: float = 1.0, name=None, verbose: bool = False):
		"""
		Create a trainable infidelity activation layer.

		Parameters
		----------
		S : float or "auto"
			Scale parameter: trainable if "auto", fixed if float in [0.01, 2.0]
		z0_init : float, "mean", or "median"
			Initial concept center value or strategy
		S_init : float
			Initial scale value when S="auto"
		name : str, optional
			Layer name for model tracking
		verbose : bool
			Enable debug output during training
		"""
		super().__init__(name, verbose=verbose)
		self.z0_init = z0_init
		self.S_init = float(S_init)
		self.S = S
		if self.S != "auto":
			self.S = float(self.S)
		self._fi_activation = Fi(z0_init=z0_init, S_init=S_init, name=name, 
		                         verbose=verbose, S=self.S)
		self.params = self._fi_activation.params
		self.grads = self._fi_activation.grads

	def forward(self, x, training=True):
		"""
		Forward pass: compute reciprocal fidelity (infidelity).

		Parameters
		----------
		x : Tensor or array-like
			Input batch

		training : bool
			If True, apply lazy initialization (if needed)

		Returns
		-------
		Tensor
			Infidelity output (reciprocal of fidelity)
		"""
		fi_value = self._fi_activation(x, training=training)
		fi_tensor = fi_value._tensor if isinstance(fi_value, Tensor) else Tensor(fi_value)._tensor
		
		# Compute reciprocal: fj = 1 / fi
		# Clamp fi to avoid division by zero
		reciprocal = 1.0 / torch.clamp(fi_tensor, min=torch.finfo(fi_tensor.dtype).eps)
		
		# Cache for inspection
		self.theta = getattr(self._fi_activation, "theta", None)
		self.out = Tensor.from_torch(reciprocal)
		
		return Tensor.from_torch(reciprocal)
