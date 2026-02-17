"""
Machine Gnostics Neural Networks (magnet)

Provides lightweight ANN, CNN, and RNN implementations with gnostic-aware
training (mg_loss, geometry E/Q, local/global GDF, scale S optimization),
following the project's regression architecture and logging style.

geometry options:
- 'E' → Estimating (Euclidian)
- 'Q' → Quantification (Minkowskian)
"""

from .base_model import ANN, CNN, RNN
from .layers.dense import Dense
from .layers.conv2d import Conv2D
from .layers.rnn import SimpleRNN
from .activations import get_activation
from .optimizers import SGD, Adam, RMSProp

__all__ = [
	'ANN', 'CNN', 'RNN',
	'Dense', 'Conv2D', 'SimpleRNN',
	'get_activation',
	'SGD', 'Adam', 'RMSProp'
]
