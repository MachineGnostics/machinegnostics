"""Dedicated public activation module for gnostic_proba.

Developer note
--------------
This module keeps the public MAGNET activation API one class per file.
"""

from __future__ import annotations

import numpy as np
import torch

from ._shared import Tensor, compute_characteristics, custom_tensor, _gnostic_activation_tensor, Layer
from .activation import Activation

class GnosticProba(Activation):
	"""Gnostic probability activation.

	This activation returns the gnostic probability characteristic and is useful
	when the model should emphasize the complementary characteristic to relevance.

	Examples
	--------
	>>> import numpy as np
	>>> from machinegnostics.magnet import Dense, GnosticProba, Sequential
	>>> model = Sequential([Dense(2, 2), GnosticProba()])
	>>> model(np.array([[0.1, 0.2]])).shape
	(1, 2)
	"""

	def __init__(self, S: float | str = 1, 
			  name=None, 
			  case:str = "i", # i or j
			  verbose: bool = False):
		"""Create a gnostic probability activation."""
		super().__init__(name, verbose=verbose)
		self.S = S
		self.case = case

	def forward(self, x, training=True):
		"""Return the gnostic probability characteristic for the supplied tensor."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		info = compute_characteristics(x.data, scale=self.S)
		self.S_local = info["S_local"]
		self.char = info['characteristics']
		self.h = np.asarray(info["hi"], dtype=np.float64)
		self.proba = self.char._idistfun(self.h)
		prime = - (4 / self.S_local) * (1 - self.proba) * self.proba
		return _gnostic_activation_tensor(x, self.proba, prime)
