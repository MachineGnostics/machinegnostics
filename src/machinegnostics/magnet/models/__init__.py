"""Model-related MAGNET components.

Developer note
--------------
Author: Nirmal Parmar, OSS Machine Gnostics

This package groups the top-level model containers used by MAGNET. Each class
now lives in its own module so the package stays easy to navigate and the root
namespace can re-export a stable public API.
"""

from .model import Model
from .sequential import Sequential
from .gnostic_neuron import GnosticNeuron

__all__ = ["Model", "Sequential", "GnosticNeuron"]