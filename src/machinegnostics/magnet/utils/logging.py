"""Logging helper facade for MAGNET.

This keeps the package using a shared logger factory without forcing callers to
depend directly on the magcal utility path.
"""

from machinegnostics.magcal.util.logging import get_logger

__all__ = ["get_logger"]