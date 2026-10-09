"""Convenience container for gnostic activation helper functions."""

from __future__ import annotations

from .ei import ei
from .fi import fi
from .fj import fj
from .hi import hi
from .hj import hj


class ActivationFunctions:
    fi = staticmethod(fi)
    fj = staticmethod(fj)
    hi = staticmethod(hi)
    hj = staticmethod(hj)
    ei = staticmethod(ei)
