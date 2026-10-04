"""Sequential model container for MAGNET.

Developer note
--------------
Author: Nirmal Parmar

This module keeps the sequential wrapper in its own file so the model package
has one dedicated class per module.
"""

from __future__ import annotations

from .model import Model


class Sequential(Model):
	"""Sequential model container for MAGNET layers.

	This is a thin semantic wrapper around :class:`Model` for layer-by-layer
	network definitions.

	Examples
	--------
	>>> from machinegnostics.magnet import Sequential
	>>> isinstance(Sequential(), Model)
	True
	"""

	pass
