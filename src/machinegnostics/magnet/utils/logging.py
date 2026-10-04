"""Logging helper facade for MAGNET.

Developer note
--------------
Author: Nirmal Parmar, OSS Machine Gnostics

This module keeps the package using a shared logger factory without forcing
callers to depend directly on the lower-level logging utility path.
"""

from machinegnostics.magcal.util.logging import get_logger

__all__ = ["get_logger"]