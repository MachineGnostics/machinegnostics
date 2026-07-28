"""Shared gnostic helpers for the magnet tensor stack."""

from __future__ import annotations

import numpy as np

from machinegnostics.magcal import GnosticsCharacteristics, ScaleParam

from .tensor import Tensor, unbroadcast

EPS = np.finfo(float).eps


def as_numpy(value) -> np.ndarray:
	if isinstance(value, Tensor):
		return value.data
	return np.asarray(value, dtype=np.float64)


def compute_characteristics(values, scale: float | str = 1.0) -> dict[str, np.ndarray | float]:
	array = as_numpy(values)
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
	info = compute_characteristics(values, scale=scale)
	fi = np.asarray(info["fi"], dtype=np.float64)
	weights = fi ** 2
	return weights / (np.sum(weights) + EPS)


def gnostic_weights_j(values, scale: float | str = 2.0):
	weights = gnostic_weights_i(values, scale=scale)
	inverse = 1.0 / (weights + EPS)
	return inverse / (np.sum(inverse) + EPS)


def custom_tensor(data, parents, backward_fn):
	requires_grad = any(getattr(parent, "requires_grad", False) for parent in parents)
	out = Tensor(data, requires_grad=requires_grad)
	out._prev = {parent for parent in parents if getattr(parent, "requires_grad", False)}
	out._backward = lambda: backward_fn(out)
	return out


def custom_tensor_from_gradient(data, parent: Tensor, gradient):
	gradient = np.asarray(gradient, dtype=np.float64)
	return custom_tensor(data, [parent], lambda out: parent._add_grad(unbroadcast(out.grad * gradient, parent.data.shape) if out.grad is not None else 0.0))
