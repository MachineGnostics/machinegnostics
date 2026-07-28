"""Loss functions for magnet."""

from __future__ import annotations

import numpy as np

from machinegnostics.magcal import GnosticsCharacteristics, ScaleParam

from ._gnostic import custom_tensor
from .tensor import Tensor


class Loss:
	def forward(self, y_pred, y_true):
		raise NotImplementedError

	def backward(self):
		raise NotImplementedError

	def __call__(self, y_pred, y_true):
		return self.forward(y_pred, y_true)


def _prepare_tensors(y_pred, y_true):
	y_pred = y_pred if isinstance(y_pred, Tensor) else Tensor(y_pred)
	y_true = y_true if isinstance(y_true, Tensor) else Tensor(y_true)
	return y_pred, y_true


class MSE(Loss):
	def forward(self, y_pred, y_true):
		y_pred, y_true = _prepare_tensors(y_pred, y_true)
		self.y_pred, self.y_true = y_pred, y_true
		return ((y_pred - y_true) ** 2).mean()

	def backward(self):
		raise NotImplementedError("MSE uses tensor autograd; call loss.backward() instead")


class BinaryCrossEntropy(Loss):
	def forward(self, y_pred, y_true, eps=1e-12):
		y_pred, y_true = _prepare_tensors(y_pred, y_true)
		self.y_pred, self.y_true = y_pred.clip(eps, 1 - eps), y_true
		return -(y_true * self.y_pred.log() + (1 - y_true) * (1 - self.y_pred).log()).mean()

	def backward(self):
		raise NotImplementedError("BinaryCrossEntropy uses tensor autograd; call loss.backward() instead")


class _BaseGnosticCharc:
	def __init__(self, S: float | str = 1):
		self.S = S

	def _get_fidelity(self, y_pred, y_true):
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
		fi = self._get_fidelity(y_pred, y_true)
		fi2 = fi ** 2
		return fi2 / np.sum(fi2 + np.finfo(float).eps)

	def _get_rentropy(self, y_pred, y_true):
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
	return custom_tensor(value, [y_pred], lambda out: y_pred._add_grad(out.grad * gradient if out.grad is not None else 0.0))


class GnosticFidelity(Loss, _BaseGnosticCharc):
	def __init__(self, S: float | str = 1):
		Loss.__init__(self)
		_BaseGnosticCharc.__init__(self, S=S)

	def forward(self, y_pred, y_true):
		y_pred, y_true = _prepare_tensors(y_pred, y_true)
		self.y_pred, self.y_true = y_pred, y_true
		fi, hi, _ = self._get_fihi(y_pred.data, y_true.data)
		return _scalar_gnostic_loss(y_pred, np.mean(fi), -(2 / self.S_local) * fi * hi)

	def backward(self):
		raise NotImplementedError("GnosticFidelity uses tensor autograd; call loss.backward() instead")


class GnosticInfidelity(Loss, _BaseGnosticCharc):
	def __init__(self, S: float | str = 1):
		Loss.__init__(self)
		_BaseGnosticCharc.__init__(self, S=S)

	def forward(self, y_pred, y_true):
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
	def __init__(self, S: float | str = 1):
		Loss.__init__(self)
		_BaseGnosticCharc.__init__(self, S=S)

	def forward(self, y_pred, y_true):
		y_pred, y_true = _prepare_tensors(y_pred, y_true)
		self.y_pred, self.y_true = y_pred, y_true
		information, fi, p = self._get_information(y_pred.data, y_true.data)
		gradient = (1 / self.S_local) * fi ** 2 * (-np.log(p) + np.log(1 - p))
		return _scalar_gnostic_loss(y_pred, np.mean(information), gradient)

	def backward(self):
		raise NotImplementedError("GnosticInformation uses tensor autograd; call loss.backward() instead")


class GnosticResidualEntropy(Loss, _BaseGnosticCharc):
	def __init__(self, S: float | str = 1):
		Loss.__init__(self)
		_BaseGnosticCharc.__init__(self, S=S)

	def forward(self, y_pred, y_true):
		y_pred, y_true = _prepare_tensors(y_pred, y_true)
		self.y_pred, self.y_true = y_pred, y_true
		rentropy, fi, hi = self._get_rentropy(y_pred.data, y_true.data)
		gradient = (2 / self.S_local) * (-fi * hi + (hi / (fi + np.finfo(float).eps)))
		return _scalar_gnostic_loss(y_pred, np.mean(rentropy), gradient)

	def backward(self):
		raise NotImplementedError("GnosticResidualEntropy uses tensor autograd; call loss.backward() instead")


class GnosticMSE(Loss, _BaseGnosticCharc):
	def __init__(self, S: float | str = 1):
		Loss.__init__(self)
		_BaseGnosticCharc.__init__(self, S=S)

	def forward(self, y_pred, y_true):
		y_pred, y_true = _prepare_tensors(y_pred, y_true)
		self.y_pred, self.y_true = y_pred, y_true
		self.gw = self._get_gw(y_pred.data, y_true.data)
		value = np.mean(self.gw * (y_pred.data - y_true.data) ** 2)
		gradient = 2.0 * self.gw * (y_pred.data - y_true.data) / y_pred.data.shape[0]
		return _scalar_gnostic_loss(y_pred, value, gradient)

	def backward(self):
		raise NotImplementedError("GnosticMSE uses tensor autograd; call loss.backward() instead")


class GnosticBinaryCrossEntropy(Loss, _BaseGnosticCharc):
	def __init__(self, S: float | str = 1):
		Loss.__init__(self)
		_BaseGnosticCharc.__init__(self, S=S)

	def forward(self, y_pred, y_true, eps=1e-12):
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
	return GnosticMSE(S=S)(y_pred, y_true)


def gnostic_weighted_rmse(y_pred, y_true, S: float | str = 1):
	value = GnosticMSE(S=S)(y_pred, y_true)
	return value ** 0.5 if isinstance(value, Tensor) else np.sqrt(value)


def fidelity_loss(y_pred, y_true, S: float | str = 1):
	return GnosticFidelity(S=S)(y_pred, y_true)


def infidelity_loss(y_pred, y_true, S: float | str = 1):
	return GnosticInfidelity(S=S)(y_pred, y_true)


def irrelevance_loss(y_pred, y_true, S: float | str = 1):
	return GnosticResidualEntropy(S=S)(y_pred, y_true)


def relevance_loss(y_pred, y_true, S: float | str = 1):
	return GnosticInformation(S=S)(y_pred, y_true)


def gnostic_characteristic_loss(y_pred, y_true, S: float | str = 1):
	return GnosticResidualEntropy(S=S)(y_pred, y_true)


def get_loss(loss):
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
