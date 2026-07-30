"""Model-related MAGNET components.

This package groups model containers and convenience neuron wrappers so future
model files can live together in one place.
"""

from .model import Model, Sequential
from .neuron import GnosticNeuron

__all__ = ["Model", "Sequential", "GnosticNeuron"]