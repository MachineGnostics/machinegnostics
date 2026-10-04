"""Public activation helper functions for MAGNET.

Developer note
--------------
This module hosts the small function-based activation helpers and the public
activation registry used by the root package export surface.
"""

from __future__ import annotations

import numpy as np

from ._shared import compute_characteristics
from .activation import Activation
from .relu import ReLU
from .step import Step
from .leakyrelu import LeakyReLU
from .elu import ELU
from .sigmoid import Sigmoid
from .softplus import Softplus
from .tanh import Tanh
from .swish import Swish
from .softmax import Softmax
from .ei import Ei
from .fi import Fi
from .fj import Fj
from .hi import Hi
from .hj import Hj
from .gnostic_proba import GnosticProba


def fi(x, S: float | str = 1):
	return np.asarray(compute_characteristics(x, scale=S)["fi"], dtype=np.float64)


def fj(x, S: float | str = 1):
	return np.asarray(compute_characteristics(x, scale=S)["fj"], dtype=np.float64)


def hi(x, S: float | str = 1):
	return np.asarray(compute_characteristics(x, scale=S)["hi"], dtype=np.float64)


def hj(x, S: float | str = 1):
	return np.asarray(compute_characteristics(x, scale=S)["hj"], dtype=np.float64)


def get_activation(activation, verbose: bool = False):
	if activation is None:
		return None
	if isinstance(activation, Activation):
		return activation
	if isinstance(activation, str):
		registry = {
			"relu": ReLU,
			"step": Step,
			"threshold": Step,
			"heaviside": Step,
			"leakyrelu": LeakyReLU,
			"elu": ELU,
			"sigmoid": Sigmoid,
			"softplus": Softplus,
			"tanh": Tanh,
			"swish": Swish,
			"softmax": Softmax,
			"fi": Fi,
			"fj": Fj,
			"hi": Hi,
			"hj": Hj,
			"ei": Ei,
			"fiactivation": Fi,
			"fjactivation": Fj,
			"hiactivation": Hi,
			"hjactivation": Hj,
			"gnosticproba": GnosticProba,
		}
		key = activation.replace("_", "").replace("-", "").lower()
		try:
			return registry[key](verbose=verbose)
		except KeyError as exc:
			raise ValueError(f"Unknown activation: {activation}") from exc
	raise TypeError(f"Unsupported activation specification: {type(activation)!r}")
