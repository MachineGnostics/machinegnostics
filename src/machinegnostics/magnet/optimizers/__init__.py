"""Optimizer exports for MAGNET (Machine Gnostics Neural Networks).

Developer note
-------------
Author: Nirmal Parmar
"""

from .base import Optimizer, get_optimizer
from .sgd import SGD
from .adam import Adam

__all__ = ["Optimizer", "get_optimizer", "SGD", "Adam"]