"""Input-weighted dense layer for MAGNET (Machine Gnostics Neural Networks).

Developer note
--------------
``iDense`` behaves like a standard dense layer, but it first applies the
gnostic estimating weights computed from the current input batch. This is the
layer to use when you want the input-side weighting path to be explicit in the
model structure.

Minimal working example
-----------------------
>>> import numpy as np
>>> from machinegnostics.magnet.layers.idense import iDense
>>> layer = iDense(2, 1)
>>> layer(np.array([[1.0, 2.0]])).shape
(1, 1)

Author: Nirmal Parmar
"""

from __future__ import annotations

from ..core._gnostic import gnostic_weights_i
from ..core.tensor import Tensor
from .dense import Dense


class iDense(Dense):
	"""Dense layer with gnostic estimating input weights.

	This layer behaves like :class:`Dense`, but before the affine transform it
	multiplies the input by the gnostic estimating weights computed from the
	current batch. That makes it useful when you want to emphasize the
	estimation-side weighting path in a compact model.

	Examples
	--------
	>>> import numpy as np
	>>> from machinegnostics.magnet import iDense
	>>> layer = iDense(2, 1)
	>>> layer(np.array([[1.0, 2.0]])).shape
	(1, 1)
	"""

	def __init__(self, in_features, out_features, weight_init=None, bias_init=None, name=None, S: float | str = 2.0, verbose: bool = False):
		"""Create an input-weighted dense layer.

		Parameters
		----------
		in_features:
			Input dimensionality.
		out_features:
			Number of output units.
		weight_init:
			Initializer for the weight matrix.
		bias_init:
			Initializer for the bias vector.
		name:
			Optional layer name.
		S:
			Scale parameter for the gnostic weighting calculation.
		verbose:
			Enable debug logging for the layer instance.

		Examples
		--------
		>>> from machinegnostics.magnet import iDense
		>>> iDense(4, 2)
		<iDense: 10 params>
		"""
		super().__init__(in_features, out_features, weight_init=weight_init, bias_init=bias_init, name=name, verbose=verbose)
		self.S = S
		self.logger.debug("iDense initialized with scale=%s.", S)

	def forward(self, x, training=True):
		"""Apply gnostic input weights before the affine transform.

		The layer first computes sample-wise estimating weights from the current
		input values, reshapes them when necessary so broadcasting is safe, and
		then hands the weighted tensor to the standard dense transform.
		"""
		x = x if isinstance(x, Tensor) else Tensor(x)
		weights = gnostic_weights_i(x.data, scale=self.S)
		if weights.shape != x.shape:
			weights = weights.reshape((1,) * (x.ndim - weights.ndim) + weights.shape)
		self.logger.debug("Running iDense forward pass with input shape %s.", x.shape)
		return super().forward(x * Tensor(weights), training=training)