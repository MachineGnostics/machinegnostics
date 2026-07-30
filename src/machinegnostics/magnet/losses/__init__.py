"""Loss functions for MAGNET (Machine Gnostics Neural Networks).

Developer note
-------------
Author: Nirmal Parmar

This module implements standard losses and gnostic losses that reuse the
Machine Gnostics characteristic engine.

Examples
--------
>>> import numpy as np
>>> from machinegnostics.magnet.losses import BinaryCrossEntropy
>>> loss = BinaryCrossEntropy()
>>> round(float(loss(np.array([[0.9]]), np.array([[1.0]]))), 3)
0.105
"""

from __future__ import annotations

import numpy as np

from machinegnostics.magcal import GnosticsCharacteristics, ScaleParam

from ..core._gnostic import custom_tensor
from ..core.tensor import Tensor


class Loss:
	"""Base class for all MAGNET losses."""

	def forward(self, y_pred, y_true):
		"""Compute a scalar loss value from predictions and targets."""
		raise NotImplementedError

	def backward(self):
		"""Return the gradient of the loss with respect to predictions."""
		raise NotImplementedError

	def __call__(self, y_pred, y_true):
		"""Shortcut for ``forward`` so losses can be called like functions."""
		return self.forward(y_pred, y_true)


def _prepare_tensors(y_pred, y_true):
	y_pred = y_pred if isinstance(y_pred, Tensor) else Tensor(y_pred)
	y_true = y_true if isinstance(y_true, Tensor) else Tensor(y_true)
	return y_pred, y_true


class MSE(Loss):
	"""Mean-squared error loss."""

	def forward(self, y_pred, y_true):
		"""Compute the average squared error."""
		y_pred, y_true = _prepare_tensors(y_pred, y_true)
		self.y_pred, self.y_true = y_pred, y_true
		return ((y_pred - y_true) ** 2).mean()

	def backward(self):
		raise NotImplementedError("MSE uses tensor autograd; call loss.backward() instead")


class BinaryCrossEntropy(Loss):
	"""Binary cross-entropy loss for sigmoid outputs."""

	def forward(self, y_pred, y_true, eps=1e-12):
		"""Compute binary cross-entropy between predictions and labels."""
		y_pred, y_true = _prepare_tensors(y_pred, y_true)
		self.y_pred, self.y_true = y_pred.clip(eps, 1 - eps), y_true
		return -(y_true * self.y_pred.log() + (1 - y_true) * (1 - self.y_pred).log()).mean()

	def backward(self):
		raise NotImplementedError("BinaryCrossEntropy uses tensor autograd; call loss.backward() instead")


class _BaseGnosticCharc:
	"""Shared helper for gnostic losses based on gnostic characteristics."""

	def __init__(self, S: float | str = 1):
		"""Store the scale parameter used by the gnostic loss family."""
		self.S = S

	def _get_fidelity(self, y_pred, y_true):
		"""Compute the fidelity characteristic for a prediction residual."""
		y_diff = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
		z_y_diff = np.exp(y_diff)
		gnostic_charc = GnosticsCharacteristics(R=z_y_diff)
		if isinstance(self.S, str) and self.S == "auto":
			scale_param = ScaleParam()
			q, q1 = gnostic_charc._get_q_q1(S=1)
			fidelity = gnostic_charc._fi(q, q1)
			self.S_local = scale_param._gscale_loc(np.mean(fidelity))
			q, q1 = gnostic_charc._get_q_q1(S=self.S_local)
			fidelity = gnostic_charc._fi(q, q1)
		else:
			self.S_local = float(self.S)
			q, q1 = gnostic_charc._get_q_q1(S=self.S_local)
			fidelity = gnostic_charc._fi(q, q1)
		return fidelity

	def _get_gw(self, y_pred, y_true):
		"""Compute normalized gnostic weights for a batch."""
		fi = self._get_fidelity(y_pred, y_true)
		fi2 = fi ** 2
		return fi2 / np.sum(fi2 + np.finfo(float).eps)

	def _get_rentropy(self, y_pred, y_true):
		"""Compute residual entropy and the auxiliary gnostic terms."""
		y_diff = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
		z_y_diff = np.exp(y_diff)
		gnostic_charc = GnosticsCharacteristics(R=z_y_diff)
		if isinstance(self.S, str) and self.S == "auto":
			scale_param = ScaleParam()
			q, q1 = gnostic_charc._get_q_q1(S=1)
			fidelity = gnostic_charc._fi(q, q1)
			self.S_local = scale_param._gscale_loc(np.mean(fidelity))
			q, q1 = gnostic_charc._get_q_q1(S=self.S_local)
			fi = gnostic_charc._fi(q, q1)
			fj = gnostic_charc._fj(q, q1)
			hi = gnostic_charc._hi(q, q1)
			rentropy = gnostic_charc._rentropy(fi, fj)
		else:
			self.S_local = float(self.S)
			q, q1 = gnostic_charc._get_q_q1(S=self.S_local)
			fi = gnostic_charc._fi(q, q1)
			fj = gnostic_charc._fj(q, q1)
			hi = gnostic_charc._hi(q, q1)
			rentropy = gnostic_charc._rentropy(fi, fj)
		return rentropy, fi, hi

	def _get_information(self, y_pred, y_true):
		"""Compute gnostic information and its auxiliary distribution terms."""
		y_diff = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
		z_y_diff = np.exp(y_diff)
		gnostic_charc = GnosticsCharacteristics(R=z_y_diff)
		if isinstance(self.S, str) and self.S == "auto":
			scale_param = ScaleParam()
			q, q1 = gnostic_charc._get_q_q1(S=1)
			fidelity = gnostic_charc._fi(q, q1)
			self.S_local = scale_param._gscale_loc(np.mean(fidelity))
			q, q1 = gnostic_charc._get_q_q1(S=self.S_local)
			fi = gnostic_charc._fi(q, q1)
			hi = gnostic_charc._hi(q, q1)
			p = gnostic_charc._idistfun(hi)
			information = gnostic_charc._info_i(p)
		else:
			self.S_local = float(self.S)
			q, q1 = gnostic_charc._get_q_q1(S=self.S_local)
			fi = gnostic_charc._fi(q, q1)
			hi = gnostic_charc._hi(q, q1)
			p = gnostic_charc._idistfun(hi)
			information = gnostic_charc._info_i(p)
		return information, fi, p

	def _get_fihi(self, y_pred, y_true):
		"""Return the fidelity, irrelevance, and effective scale value."""
		y_diff = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
		z_y_diff = np.exp(y_diff)
		gnostic_charc = GnosticsCharacteristics(R=z_y_diff)
		if isinstance(self.S, str) and self.S == "auto":
			scale_param = ScaleParam()
			q, q1 = gnostic_charc._get_q_q1(S=1)
			fi = gnostic_charc._fi(q, q1)
			self.S_local = scale_param._gscale_loc(np.mean(fi))
			q, q1 = gnostic_charc._get_q_q1(S=self.S_local)
			fi = gnostic_charc._fi(q, q1)
			hi = gnostic_charc._hi(q, q1)
		else:
			self.S_local = float(self.S)
			q, q1 = gnostic_charc._get_q_q1(S=self.S_local)
			fi = gnostic_charc._fi(q, q1)
			hi = gnostic_charc._hi(q, q1)
		return fi, hi, self.S_local


def _scalar_gnostic_loss(y_pred, value, gradient):
	"""Wrap a scalar gnostic quantity in a differentiable tensor."""
	return custom_tensor(value, [y_pred], lambda out: y_pred._add_grad(out.grad * gradient if out.grad is not None else 0.0))


class GnosticFidelity(Loss, _BaseGnosticCharc):
	"""Loss that minimizes the mean gnostic fidelity residual term."""

	def __init__(self, S: float | str = 1):
		"""Create a fidelity-based gnostic loss."""
		Loss.__init__(self)
		_BaseGnosticCharc.__init__(self, S=S)

	def forward(self, y_pred, y_true):
		"""Compute the fidelity-based gnostic loss."""
		y_pred, y_true = _prepare_tensors(y_pred, y_true)
		self.y_pred, self.y_true = y_pred, y_true
		fi, hi, _ = self._get_fihi(y_pred.data, y_true.data)
		return _scalar_gnostic_loss(y_pred, np.mean(fi), -(2 / self.S_local) * fi * hi)

	def backward(self):
		raise NotImplementedError("GnosticFidelity uses tensor autograd; call loss.backward() instead")


class GnosticInfidelity(Loss, _BaseGnosticCharc):
	"""Loss that emphasizes the gnostic infidelity characteristic."""

	def __init__(self, S: float | str = 1):
		"""Create an infidelity-based gnostic loss."""
		Loss.__init__(self)
		_BaseGnosticCharc.__init__(self, S=S)

	def forward(self, y_pred, y_true):
		"""Compute the infidelity-based gnostic loss."""
		y_pred, y_true = _prepare_tensors(y_pred, y_true)
		self.y_pred, self.y_true = y_pred, y_true
		fi, hi, _ = self._get_fihi(y_pred.data, y_true.data)
		value = np.mean(1.0 / (fi + 1e-21))
		gradient = (2 / self.S_local) * (hi / (fi + np.finfo(float).eps))
		gradient = np.clip(gradient, -1e12, 1e12)
		return _scalar_gnostic_loss(y_pred, value, gradient)

	def backward(self):
		raise NotImplementedError("GnosticInfidelity uses tensor autograd; call loss.backward() instead")


class GnosticInformation(Loss, _BaseGnosticCharc):
	"""Loss that measures gnostic information content."""

	def __init__(self, S: float | str = 1):
		"""Create an information-based gnostic loss."""
		Loss.__init__(self)
		_BaseGnosticCharc.__init__(self, S=S)

	def forward(self, y_pred, y_true):
		"""Compute the gnostic information loss."""
		y_pred, y_true = _prepare_tensors(y_pred, y_true)
		self.y_pred, self.y_true = y_pred, y_true
		information, fi, p = self._get_information(y_pred.data, y_true.data)
		gradient = (1 / self.S_local) * fi ** 2 * (-np.log(p) + np.log(1 - p))
		return _scalar_gnostic_loss(y_pred, np.mean(information), gradient)

	def backward(self):
		raise NotImplementedError("GnosticInformation uses tensor autograd; call loss.backward() instead")


class GnosticResidualEntropy(Loss, _BaseGnosticCharc):
	"""Loss based on residual entropy from gnostic characteristics."""

	def __init__(self, S: float | str = 1):
		"""Create a residual-entropy gnostic loss."""
		Loss.__init__(self)
		_BaseGnosticCharc.__init__(self, S=S)

	def forward(self, y_pred, y_true):
		"""Compute the residual-entropy-based gnostic loss."""
		y_pred, y_true = _prepare_tensors(y_pred, y_true)
		self.y_pred, self.y_true = y_pred, y_true
		rentropy, fi, hi = self._get_rentropy(y_pred.data, y_true.data)
		gradient = (2 / self.S_local) * (-fi * hi + (hi / (fi + np.finfo(float).eps)))
		return _scalar_gnostic_loss(y_pred, np.mean(rentropy), gradient)

	def backward(self):
		raise NotImplementedError("GnosticResidualEntropy uses tensor autograd; call loss.backward() instead")


class GnosticMSE(Loss, _BaseGnosticCharc):
	"""Weighted mean-squared error using gnostic weights."""

	def __init__(self, S: float | str = 1):
		"""Create a gnostic MSE loss."""
		Loss.__init__(self)
		_BaseGnosticCharc.__init__(self, S=S)

	def forward(self, y_pred, y_true):
		"""Compute gnostic weighted MSE."""
		y_pred, y_true = _prepare_tensors(y_pred, y_true)
		self.y_pred, self.y_true = y_pred, y_true
		self.gw = self._get_gw(y_pred.data, y_true.data)
		value = np.mean(self.gw * (y_pred.data - y_true.data) ** 2)
		gradient = 2.0 * self.gw * (y_pred.data - y_true.data) / y_pred.data.shape[0]
		return _scalar_gnostic_loss(y_pred, value, gradient)

	def backward(self):
		raise NotImplementedError("GnosticMSE uses tensor autograd; call loss.backward() instead")


class GnosticBinaryCrossEntropy(Loss, _BaseGnosticCharc):
	"""Weighted binary cross-entropy using gnostic weights."""

	def __init__(self, S: float | str = 1):
		"""Create a gnostic BCE loss."""
		Loss.__init__(self)
		_BaseGnosticCharc.__init__(self, S=S)

	def forward(self, y_pred, y_true, eps=1e-12):
		"""Compute gnostic weighted binary cross-entropy."""
		y_pred, y_true = _prepare_tensors(y_pred, y_true)
		self.y_pred = y_pred.clip(eps, 1 - eps)
		self.y_true = y_true
		self.gw = self._get_gw(y_pred.data, y_true.data)
		value = -np.mean(self.gw * (y_true.data * np.log(self.y_pred.data) + (1 - y_true.data) * np.log(1 - self.y_pred.data)))
		gradient = self.gw * (self.y_pred.data - y_true.data) / (self.y_pred.data * (1 - self.y_pred.data) * y_pred.data.shape[0])
		return _scalar_gnostic_loss(self.y_pred, value, gradient)

	def backward(self):
		raise NotImplementedError("GnosticBinaryCrossEntropy uses tensor autograd; call loss.backward() instead")


def gnostic_weighted_mse(y_pred, y_true, S: float | str = 1):
	"""Convenience wrapper for gnostic weighted MSE."""
	return GnosticMSE(S=S)(y_pred, y_true)


def gnostic_weighted_rmse(y_pred, y_true, S: float | str = 1):
	"""Convenience wrapper for gnostic weighted RMSE."""
	value = GnosticMSE(S=S)(y_pred, y_true)
	return value ** 0.5 if isinstance(value, Tensor) else np.sqrt(value)


def fidelity_loss(y_pred, y_true, S: float | str = 1):
	"""Convenience wrapper for the fidelity-based gnostic loss."""
	return GnosticFidelity(S=S)(y_pred, y_true)


def infidelity_loss(y_pred, y_true, S: float | str = 1):
	"""Convenience wrapper for the infidelity-based gnostic loss."""
	return GnosticInfidelity(S=S)(y_pred, y_true)


def irrelevance_loss(y_pred, y_true, S: float | str = 1):
	"""Convenience wrapper for the residual-entropy gnostic loss."""
	return GnosticResidualEntropy(S=S)(y_pred, y_true)


def relevance_loss(y_pred, y_true, S: float | str = 1):
	"""Convenience wrapper for the gnostic information loss."""
	return GnosticInformation(S=S)(y_pred, y_true)


def gnostic_characteristic_loss(y_pred, y_true, S: float | str = 1):
	"""Alias for the gnostic residual-entropy loss."""
	return GnosticResidualEntropy(S=S)(y_pred, y_true)


def get_loss(loss):
	"""Resolve a string or loss instance into a callable loss object.

	Examples
	--------
	>>> from machinegnostics.magnet.losses import get_loss
	>>> get_loss("mse")
	MSE()
	"""
	if loss is None:
		return MSE()
	if isinstance(loss, Loss):
		return loss
	if isinstance(loss, str):
		registry = {
			"mse": MSE(),
			"binarycrossentropy": BinaryCrossEntropy(),
			"bce": BinaryCrossEntropy(),
			"gnosticmse": GnosticMSE(),
			"gnosticbinarycrossentropy": GnosticBinaryCrossEntropy(),
			"gnosticfidelity": GnosticFidelity(),
			"gnosticinfidelity": GnosticInfidelity(),
			"gnosticinformation": GnosticInformation(),
			"gnosticresidualentropy": GnosticResidualEntropy(),
		}
		key = loss.replace("_", "").replace("-", "").lower()
		try:
			return registry[key]
		except KeyError as exc:
			raise ValueError(f"Unknown loss: {loss}") from exc
	raise TypeError(f"Unsupported loss specification: {type(loss)!r}")
