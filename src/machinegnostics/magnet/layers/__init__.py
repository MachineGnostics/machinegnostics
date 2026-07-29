"""Layer exports for MAGNET (Machine Gnostics Neural Networks).

Developer note
-------------
Author: Nirmal Parmar
"""

from .base import Layer
from .dense import Dense
from .idense import iDense
from .jdense import jDense
from .batchnorm import BatchNorm, GnosticBatchNorm
from .flatten import Flatten

__all__ = ["Layer", "Dense", "iDense", "jDense", "BatchNorm", "GnosticBatchNorm", "Flatten"]