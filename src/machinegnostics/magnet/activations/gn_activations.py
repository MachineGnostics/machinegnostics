"""Gnostic activation helper container.

Developer note
--------------
Author: Nirmal Parmar, OSS Machine Gnostics

This module provides a small compatibility wrapper exposing the gnostic
characteristic helper functions as static methods for older call patterns.
"""

from __future__ import annotations

from . import fi, fj, hi, hj


class ActivationFunctions:
	"""Container object that mirrors the public gnostic activation helpers."""

	fi = staticmethod(fi)
	fj = staticmethod(fj)
	hi = staticmethod(hi)
	hj = staticmethod(hj)