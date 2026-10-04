"""Dedicated public activation module for ei.

Developer note
--------------
This module keeps the public MAGNET activation API one class per file.
"""

from __future__ import annotations

import numpy as np
import torch

from ._shared import Tensor, compute_characteristics, custom_tensor, _gnostic_activation_tensor, Layer
from .activation import Activation

class Ei(Activation):
	"""Gnostic entropy complement activation.

	This activation returns ``1 - Fi`` for the supplied tensor, giving users a
	direct entropy-style signal derived from the fidelity characteristic.

	Examples
	--------
	>>> import numpy as np
	>>> from machinegnostics.magnet import Ei
	>>> Ei()(np.array([0.1, 0.2])).shape
	(2,)
	"""

	def __init__(self, S: float | str = 1, name=None, verbose: bool = False):
		"""Create an Ei activation."""
		super().__init__(name, verbose=verbose)
		self.S = S

	def forward(self, x, training=True):
		"""Return the complement of the fidelity characteristic for the input."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		info = compute_characteristics(x.data, scale=self.S)
		self.S_local = info["S_local"]
		self.fidelity = np.asarray(info["fi"], dtype=np.float64)
		self.hi = np.asarray(info["hi"], dtype=np.float64)
		self.entropy = 1.0 - self.fidelity
		prime = (2.0 * self.fidelity * self.hi + np.finfo(float).eps) / self.S_local
		return _gnostic_activation_tensor(x, self.entropy, prime)
