"""Activation layers and gnostic characteristic helpers for MAGNET.

Developer note
--------------
Author: Nirmal Parmar

This package now keeps one public class per file and re-exports the MAGNET
activation surface from here.
"""

from .helpers import fi, fj, hi, hj, get_activation
from .activation import Activation
from .relu import ReLU
from .sigmoid import Sigmoid
from .tanh import Tanh
from .softmax import Softmax
from .step import Step
from .leakyrelu import LeakyReLU
from .elu import ELU
from .softplus import Softplus
from .swish import Swish
from .ei import Ei
from .fi import Fi
from .fj import Fj
from .hi import Hi
from .hj import Hj
from .gnostic_proba import GnosticProba
from .square import Square
from .gn_activations import ActivationFunctions

__all__ = [
	"fi", "fj", "hi", "hj", "get_activation",
	"Activation", "ReLU", "Sigmoid", "Tanh", "Softmax", "Step", "LeakyReLU", "ELU", "Softplus", "Swish",
	"Ei", "Fi", "Fj", "Hi", "Hj", "GnosticProba", "Square", "ActivationFunctions",
]
