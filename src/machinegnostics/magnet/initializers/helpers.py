"""Public initializer helper functions for MAGNET.

Developer note
--------------
Author: Nirmal Parmar, OSS Machine Gnostics

This module resolves initializer aliases used by the flat MAGNET API and
returns the concrete initializer instance requested by the caller.
"""

from __future__ import annotations

from .base import Initializer, _compute_fans
from .zeros import Zeros
from .ones import Ones
from .random_normal import RandomNormal
from .xavier_uniform import XavierUniform
from .he_normal import HeNormal


def get_initializer(initializer):
	"""Resolve an initializer specification into an initializer instance."""
	if initializer is None:
		return XavierUniform()
	if isinstance(initializer, Initializer):
		return initializer
	if isinstance(initializer, str):
		registry = {
			"zeros": Zeros(),
			"ones": Ones(),
			"randomnormal": RandomNormal(),
			"xavieruniform": XavierUniform(),
			"henormal": HeNormal(),
		}
		key = initializer.replace("_", "").replace("-", "").lower()
		try:
			return registry[key]
		except KeyError as exc:
			raise ValueError(f"Unknown initializer: {initializer}") from exc
	raise TypeError(f"Unsupported initializer specification: {type(initializer)!r}")
