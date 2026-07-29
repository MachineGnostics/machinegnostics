"""Input-weighted dense layer for MAGNET (Machine Gnostics Neural Networks).

Developer note
--------------
``jDense`` mirrors :class:`iDense` but uses the gnostic quantifying weights.
It is useful when the model should emphasize the quantification-side weighting
path before the affine transform.

Minimal working example
-----------------------
>>> import numpy as np
>>> from machinegnostics.magnet import jDense
>>> layer = jDense(2, 1)
>>> layer(np.array([[1.0, 2.0]])).shape
(1, 1)

Author: Nirmal Parmar
"""

from __future__ import annotations

from .._gnostic import gnostic_weights_j
from ..tensor import Tensor
from .dense import Dense


class jDense(Dense):
	"""Dense layer with gnostic quantifying input weights.

	This layer mirrors :class:`iDense`, but it applies the gnostic quantifying
	weights before the dense transform. Use it when the quantification path is
	more meaningful for the model you are building.

	Examples
	--------
	>>> import numpy as np
	>>> from machinegnostics.magnet import jDense
	>>> layer = jDense(2, 1)
	>>> layer(np.array([[1.0, 2.0]])).shape
	(1, 1)
	"""

	def __init__(self, in_features, out_features, weight_init=None, bias_init=None, name=None, S: float | str = 2.0):
		"""Create a quantifying-weight dense layer.

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
		"""
		super().__init__(in_features, out_features, weight_init=weight_init, bias_init=bias_init, name=name)
		self.S = S
		self.logger.debug("jDense initialized with scale=%s.", S)

	def forward(self, x, training=True):
		"""Apply gnostic quantifying weights before the affine transform.

		The layer computes quantifying weights from the current batch, reshapes
		them for broadcasting when needed, and then passes the weighted input to
		the dense affine transform.
		"""
		x = x if isinstance(x, Tensor) else Tensor(x)
		weights = gnostic_weights_j(x.data, scale=self.S)
		if weights.shape != x.shape:
			weights = weights.reshape((1,) * (x.ndim - weights.ndim) + weights.shape)
		self.logger.debug("Running jDense forward pass with input shape %s.", x.shape)
		return super().forward(x * Tensor(weights), training=training)