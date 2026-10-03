"""Backward-compatible compatibility imports for MAGNET model containers.

Older MAGNET code sometimes imported ``Model`` and ``Sequential`` from
``machinegnostics.magnet.models.model``. The canonical implementations
now live in ``base.py`` and ``sequential.py``, but this shim stays in
place so those historical import paths continue to work.
"""

from .base import Model
from .sequential import Sequential

__all__ = ['Model', 'Sequential']
