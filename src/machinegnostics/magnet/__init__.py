"""Public API for the lightweight Magnet backend.

This package now exposes only actively maintained, numpy-native building blocks.
Legacy npnet-era modules were removed to keep the surface minimal and stable.
"""

from . import initializers as init
from . import activations as act
from . import metrics

from .base_model import ANN, Sequential
from .keras_layers import Dense
from .optimizers import Adam, SGD