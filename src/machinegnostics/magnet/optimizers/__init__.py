"""Optimizer exports for MAGNET (Machine Gnostics Neural Networks).

Developer note
-------------
Author: Nirmal Parmar

Examples
--------
>>> from machinegnostics.magnet import Adam, SGD
>>> Adam(lr=0.001)
Adam(...)
>>> SGD(lr=0.01)
SGD(...)
"""

from .base import Optimizer, get_optimizer
from .adagrad import Adagrad
from .sgd import SGD
from .adam import Adam
from .rmsprop import RMSprop

__all__ = ["Optimizer", "get_optimizer", "Adagrad", "SGD", "Adam", "RMSprop"]