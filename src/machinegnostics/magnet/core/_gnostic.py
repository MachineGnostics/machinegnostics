"""Shared gnostic helpers for MAGNET.

Developer note
-------------
Author: Nirmal Parmar, Machine Gnostics

This module keeps the gnostic characteristic math separate from the public
model API. The characteristic calculations still use NumPy because they are
domain-specific and already live in the Machine Gnostics ecosystem, but the
gradient bridge is now torch-based so the rest of MAGNET can train on hidden
PyTorch tensors.

Bird's-eye view
---------------
- ``compute_characteristics`` produces the gnostic characteristics in NumPy.
- ``custom_tensor`` bridges NumPy values back into the torch autograd graph.
- ``gnostic_weights_i`` and ``gnostic_weights_j`` keep the current weighting
  semantics used by the gnostic layers.
"""

from __future__ import annotations

import numpy as np
import torch

from machinegnostics.magcal import GnosticsCharacteristics, ScaleParam

from ..utils.logging import get_logger
from .tensor import Tensor, to_numpy

EPS = np.finfo(float).eps
logger = get_logger(__name__)


def as_numpy(value) -> np.ndarray:
	"""Return a NumPy view for a tensor or array-like input."""
	return to_numpy(value)


def compute_characteristics(values, scale: float | str = 1.0) -> dict[str, np.ndarray | float]:
	"""Compute the shared gnostic characteristics for a value tensor."""
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


class _GnosticBridge(torch.autograd.Function):
	@staticmethod
	def forward(ctx, parent_tensor: torch.Tensor, value_tensor: torch.Tensor, gradient_tensor: torch.Tensor):
		ctx.save_for_backward(gradient_tensor)
		return value_tensor

	@staticmethod
	def backward(ctx, grad_output):
		(gradient_tensor,) = ctx.saved_tensors
		return grad_output * gradient_tensor, None, None


def _prepare_bridge_tensors(parent: Tensor, data, gradient):
	value_tensor = torch.as_tensor(np.asarray(data, dtype=np.float64), device=parent._tensor.device, dtype=parent._tensor.dtype)
	gradient_tensor = torch.as_tensor(np.asarray(gradient, dtype=np.float64), device=parent._tensor.device, dtype=parent._tensor.dtype)
	return value_tensor, gradient_tensor


def custom_tensor(data, parent: Tensor | list[Tensor] | tuple[Tensor, ...], gradient):
	"""Create a differentiable tensor using a custom gnostic gradient."""
	if isinstance(parent, (list, tuple)):
		if len(parent) != 1:
			raise ValueError("MAGNET gnostic bridge currently supports exactly one parent tensor")
		parent = parent[0]
	if not isinstance(parent, Tensor):
		raise TypeError("parent must be a MAGNET Tensor")
	value_tensor, gradient_tensor = _prepare_bridge_tensors(parent, data, gradient)
	bridged = _GnosticBridge.apply(parent._tensor, value_tensor, gradient_tensor)
	return Tensor.from_torch(bridged)


def custom_tensor_from_gradient(data, parent: Tensor, gradient):
	"""Create a tensor whose gradient is scaled by a fixed factor."""
	return custom_tensor(data, parent, gradient)
