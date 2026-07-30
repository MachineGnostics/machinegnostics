"""Shared gnostic helpers for MAGNET (Machine Gnostics Neural Networks).

Developer note
-------------
Author: Nirmal Parmar

This module centralizes gnostic characteristic calculations used by activations,
losses, and gnostic-weighted layers inside MAGNET.

Examples
--------
>>> import numpy as np
>>> from machinegnostics.magnet._gnostic import gnostic_weights_i
>>> gnostic_weights_i(np.array([[0.1, 0.2], [0.2, 0.3]])).shape
(2, 2)
"""

from __future__ import annotations

import numpy as np

from machinegnostics.magcal import GnosticsCharacteristics, ScaleParam

from .tensor import Tensor, unbroadcast
from ..utils.logging import get_logger

EPS = np.finfo(float).eps
logger = get_logger(__name__)


def as_numpy(value) -> np.ndarray:
	"""Return a NumPy view for a tensor or array-like input.

	Parameters
	----------
	value:
		Tensor or array-like object.

	Returns
	-------
	numpy.ndarray
		A float64 NumPy array.
	"""
	if isinstance(value, Tensor):
		logger.debug("Converted Tensor to NumPy array with shape %s.", value.data.shape)
		return value.data
	logger.debug("Converted array-like value to NumPy array.")
	return np.asarray(value, dtype=np.float64)


def compute_characteristics(values, scale: float | str = 1.0) -> dict[str, np.ndarray | float]:
	"""Compute the shared gnostic characteristics for a value tensor.

	Examples
	--------
	>>> import numpy as np
	>>> info = compute_characteristics(np.array([0.1, 0.5, 1.0]))
	>>> sorted(info.keys())
	['S_local', 'characteristics', 'fi', 'fj', 'hi', 'hj']
	"""
	array = as_numpy(values)
	logger.debug("Computing gnostic characteristics for array shape %s with scale=%s.", array.shape, scale)
	z_values = np.exp(array - np.median(array))
	characteristics = GnosticsCharacteristics(R=z_values)
	if isinstance(scale, str) and scale == "auto":
		q, q1 = characteristics._get_q_q1(S=1)
		fi_seed = characteristics._fi(q, q1)
		local_scale = ScaleParam()._gscale_loc(np.mean(fi_seed))
	else:
		local_scale = float(scale)
	q, q1 = characteristics._get_q_q1(S=local_scale)
	fi = characteristics._fi(q, q1)
	fj = characteristics._fj(q, q1)
	hi = characteristics._hi(q, q1)
	hj = characteristics._hj(q, q1)
	return {
		"S_local": local_scale,
		"fi": fi,
		"fj": fj,
		"hi": hi,
		"hj": hj,
		"characteristics": characteristics,
	}


def gnostic_weights_i(values, scale: float | str = 2.0):
	"""Return normalized estimating weights for MAGNET layers."""
	info = compute_characteristics(values, scale=scale)
	fi = np.asarray(info["fi"], dtype=np.float64)
	weights = fi ** 2
	logger.debug("Computed gnostic i-weights with shape %s.", weights.shape)
	return weights / (np.sum(weights) + EPS)


def gnostic_weights_j(values, scale: float | str = 2.0):
	"""Return normalized quantifying weights for MAGNET layers."""
	weights = gnostic_weights_i(values, scale=scale)
	inverse = 1.0 / (weights + EPS)
	logger.debug("Computed gnostic j-weights with shape %s.", inverse.shape)
	return inverse / (np.sum(inverse) + EPS)


def custom_tensor(data, parents, backward_fn):
	"""Create a Tensor whose backward pass is defined by a closure."""
	requires_grad = any(getattr(parent, "requires_grad", False) for parent in parents)
	logger.debug("Creating custom tensor with %s parents and requires_grad=%s.", len(parents), requires_grad)
	out = Tensor(data, requires_grad=requires_grad)
	out._prev = {parent for parent in parents if getattr(parent, "requires_grad", False)}
	out._backward = lambda: backward_fn(out)
	return out


def custom_tensor_from_gradient(data, parent: Tensor, gradient):
	"""Create a tensor whose gradient is scaled by a fixed factor.

	This is useful for wrapping gnostic scalar outputs where the analytic gradient
	is known ahead of time.
	"""
	gradient = np.asarray(gradient, dtype=np.float64)
	logger.debug("Creating custom tensor from fixed gradient with shape %s.", gradient.shape)
	return custom_tensor(data, [parent], lambda out: parent._add_grad(unbroadcast(out.grad * gradient, parent.data.shape) if out.grad is not None else 0.0))
