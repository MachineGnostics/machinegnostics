"""Dedicated public activation module for hj.

Developer note
--------------
This module keeps the public MAGNET activation API one class per file.
"""

from __future__ import annotations

import numpy as np
import torch

from ._shared import Tensor, compute_characteristics, custom_tensor, _gnostic_activation_tensor, Layer
from .activation import Activation
from ._centered import _CenteredCharacteristicActivation

class Hj(_CenteredCharacteristicActivation):
	"""Trainable irrelevance activation inspired by ``hj``.

	This layer learns a concept center ``z0`` and uses a bounded scale ``S`` to
	produce a centered irrelevance response. When ``S`` is not supplied, the
	layer keeps the existing bounded trainable-scale behavior; when ``S`` is
	provided, it is fixed and only ``z0`` is learned.

	Examples
	--------
	>>> import numpy as np
	>>> from machinegnostics.magnet import Dense, Hj, Sequential
	>>> model = Sequential([Dense(2, 2), Hj()])
	>>> model(np.array([[0.1, 0.2]])).shape
	(1, 2)
	"""

	def _transform(self, theta: torch.Tensor) -> torch.Tensor:
		return torch.sinh(torch.clamp(2.0 * theta, -20.0, 20.0))
