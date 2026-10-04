"""Public loss helper functions for MAGNET.

Developer note
--------------
Author: Nirmal Parmar, OSS Machine Gnostics

This module provides the user-facing convenience helpers for the MAGNET loss
registry. Each helper is a thin constructor/call wrapper around the dedicated
loss class so that callers can keep using the flat package API.
"""

from __future__ import annotations

import numpy as np

from .binary_crossentropy import BinaryCrossEntropy
from .mse import MSE
from .gnostic_fidelity_loss import FidelityLoss
from .gnostic_infidelity_loss import InfidelityLoss
from .gnostic_information_loss import InformationLoss
from .residual_entropy_loss import ResidualEntropyLoss
from .gnostic_mse_loss import MSELoss
from .gnostic_binary_crossentropy_loss import BinaryCrossEntropyLoss
from .gnostic_rss_loss import RSSLoss
from .gnostic_iss_loss import ISSLoss


def gnostic_weighted_mse(y_pred, y_true, S: float | str = 1):
	"""Return the gnostic weighted mean-squared error for two tensors."""
	return MSELoss(S=S)(y_pred, y_true)


def gnostic_weighted_rmse(y_pred, y_true, S: float | str = 1):
	"""Return the root mean-squared gnostic weighted error."""
	value = MSELoss(S=S)(y_pred, y_true)
	return value ** 0.5 if hasattr(value, '__pow__') else np.sqrt(value)


def fidelity_loss(y_pred, y_true, S: float | str = 1):
	"""Return the fidelity-oriented gnostic loss."""
	return FidelityLoss(S=S)(y_pred, y_true)


def infidelity_loss(y_pred, y_true, S: float | str = 1):
	"""Return the infidelity-oriented gnostic loss."""
	return InfidelityLoss(S=S)(y_pred, y_true)


def irrelevance_loss(y_pred, y_true, S: float | str = 1):
	"""Return the residual-entropy loss used for irrelevance modeling."""
	return ResidualEntropyLoss(S=S)(y_pred, y_true)


def relevance_loss(y_pred, y_true, S: float | str = 1):
	"""Return the information-oriented gnostic loss."""
	return InformationLoss(S=S)(y_pred, y_true)


def gnostic_characteristic_loss(y_pred, y_true, S: float | str = 1):
	"""Return the residual-entropy gnostic characteristic loss."""
	return ResidualEntropyLoss(S=S)(y_pred, y_true)


def get_loss(loss):
	"""Resolve a loss specification into a loss instance.

	Parameters
	----------
	loss:
		``None``, a loss instance, or a string alias understood by the MAGNET
		loss registry.

	Returns
	-------
	Loss
		The resolved loss instance.
	"""
	if loss is None:
		return MSE()
	if isinstance(loss, (MSE, BinaryCrossEntropy, FidelityLoss, InfidelityLoss, RSSLoss, ISSLoss, InformationLoss, ResidualEntropyLoss, MSELoss, BinaryCrossEntropyLoss)):
		return loss
	if isinstance(loss, str):
		registry = {
			"mse": MSE(),
			"binarycrossentropy": BinaryCrossEntropy(),
			"bce": BinaryCrossEntropy(),
			"mseloss": MSELoss(),
			"binarycrossentropyloss": BinaryCrossEntropyLoss(),
			"fidelityloss": FidelityLoss(),
			"infidelityloss": InfidelityLoss(),
			"informationloss": InformationLoss(),
			"residualentropyloss": ResidualEntropyLoss(),
			"rssloss": RSSLoss(),
			"issloss": ISSLoss(),
			"fidelity": FidelityLoss(),
			"infidelity": InfidelityLoss(),
			"information": InformationLoss(),
			"residualentropy": ResidualEntropyLoss(),
			"rss": RSSLoss(),
			"iss": ISSLoss(),
		}
		key = loss.replace('_', '').replace('-', '').lower()
		try:
			return registry[key]
		except KeyError as exc:
			raise ValueError(f'Unknown loss: {loss}') from exc
	raise TypeError(f'Unsupported loss specification: {type(loss)!r}')
