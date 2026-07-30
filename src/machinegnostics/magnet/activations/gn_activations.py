"""Gnostic activation helper container."""

from __future__ import annotations

from . import fi, fj, hi, hj


class ActivationFunctions:
	fi = staticmethod(fi)
	fj = staticmethod(fj)
	hi = staticmethod(hi)
	hj = staticmethod(hj)