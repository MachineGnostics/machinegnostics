"""Sequential model container for MAGNET.

Developer note
--------------
Author: Nirmal Parmar

``Sequential`` is the user-facing container for stacking MAGNET layers in
order. It inherits all training behavior from ``Model`` and keeps the
semantics familiar: the output of one layer becomes the input of the next.
"""

from __future__ import annotations

from typing import Iterable

from ..layers.base import Layer
from .base import Model


class Sequential(Model):
	"""Sequential model: layers applied one after another.

	This class is a thin semantic wrapper around ``Model`` for layer-by-layer
	network definitions. It keeps the inherited compile/fit/evaluate/predict
	behavior while giving users a dedicated sequential container that matches the
	project's one-public-class-per-file standard.

	Examples
	--------
	>>> from machinegnostics.magnet import Dense, Sequential
	>>> model = Sequential([Dense(2, 4), Dense(4, 1)])
	>>> len(model.layers)
	2
	"""

	def __init__(self, layers: Iterable[Layer] | None = None, verbose: bool = False):
		super().__init__(layers=layers, verbose=verbose)
