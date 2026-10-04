"""Dedicated public module for Optimizer.

Developer note
--------------
This module keeps the public MAGNET API organized one class per file while
reusing the existing shared implementation from the package initializer.
"""

from .__init__ import Optimizer

__all__ = ["Optimizer"]
